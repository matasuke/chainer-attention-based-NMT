"""main training loop for seq2seq"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')

from chainer.dataset import convert
import chainer
from chainer.backends import cuda
from chainer.training import extensions
from chainer import training
from chainer import serializers

from Seq2SeqDataset import Seq2SeqDatasetBase
from common.record import record_settings
from common.ENV import SLACK_URL, SLACK_REPORT_CHANNEL_NAME, SLACK_TRANSLATION_CHANNEL_NAME
from extensions.SlackNortifier import SlackNortifier, post2slack
from extensions.CalculateBleu import CalculateBleu
from net import seq2seq


def get_arguments():
    """helper to get arguments"""

    batchsize = 64
    epoch = 30
    unit_size = 512
    num_layer = 3
    source_min_token = 1
    source_max_token = 50
    target_min_token = 1
    target_max_token = 50
    log_interval = 1
    validation_interval = 10
    out_dir = 'result'
    nortify_to_slack = False

    parser = argparse.ArgumentParser()
    parser.add_argument("SOURCE", type=str,
                        help="path to source dataset")
    parser.add_argument('TARGET', type=str,
                        help='path to target dataset')
    parser.add_argument('SOURCE_VOCAB', type=str,
                        help="path to source vocaburaly dict")
    parser.add_argument('TARGET_VOCAB', type=str,
                        help="path to target vocaburaly dict")
    parser.add_argument("--mode", metavar='INT', type=str,
                        default='train', choices=['train', 'reuse'],
                        help="type of using model, 'train', 'reuse'")
    parser.add_argument('--validation_source', metavar='str', type=str, default='',
                        help='path to validation source dataste')
    parser.add_argument('--validation_target', metavar='str', type=str, default='',
                        help='path to validation target dataset')
    parser.add_argument('--batchsize', '-b', metavar='INT', type=int, default=batchsize,
                        help="minibatch size (default: %(default)d")
    parser.add_argument('--epoch', '-e', metavar='INT', type=int, default=epoch,
                        help="number of training epoch (default: %(default)d")
    parser.add_argument('--gpu', '-g', metavar="INT", type=int, default=-1,
                        help="GPU ID (default: %(default)d")
    parser.add_argument('--unit', '-u', metavar='INT', type=int, default=unit_size,
                        help="hidden layer size (defalt: %(default)d)")
    parser.add_argument('--layer', '-l', metavar='INT', type=int, default=num_layer,
                        help="number of layers (default: %(default)d)")
    parser.add_argument('--source_min_token', metavar='INT', type=int, default=source_min_token,
                        help="number of min tokens in source sentences (defalt: %(default)d)")
    parser.add_argument('--source_max_token', metavar='INT', type=int, default=source_max_token,
                        help="number of max tokens in source sentences (defalt: %(default)d)")
    parser.add_argument('--target_min_token', metavar='INT', type=int, default=target_min_token,
                        help="number of min tokens in target sentences (defalt: %(default)d)")
    parser.add_argument('--target_max_token', metavar='INT', type=int, default=target_max_token,
                        help="number of max tokens in target sentences (defalt: %(default)d)")
    parser.add_argument('--log_interval', metavar='INT', type=int, default=log_interval,
                        help="number of iteration to show log (defalt: %(default)d~")
    parser.add_argument('--validation_interval', metavar='INT',
                        type=int, default=validation_interval,
                        help="number of iteration to evaluate the model (defalt: %(default)d)")
    parser.add_argument('--out', '-o', metavar='str', type=str, default=out_dir,
                        help="directory to output the result (defalt: %(default)s)")
    parser.add_argument('--slack', action='store_true', default=nortify_to_slack,
                        help="Report training result to Slack (defalt: %(default)s)\
                              If you want to use this options. you have to set Environment Variable\
                              written in src/common/ENV.py")
    args = parser.parse_args()

    try:
        if not Path(args.SOURCE).exists():
            msg = "source dataset file %s is not found" % args.SOURCE
            raise FileNotFoundError(msg)
        if not Path(args.TARGET).exists():
            msg = "target datast file %s is not found" % args.TARGET
            raise FileNotFoundError(msg)
        if not Path(args.SOURCE_VOCAB).exists():
            msg = "source vocab file %s is not found" % args.SOURCE_VOCAB
            raise FileNotFoundError(msg)
        if not Path(args.TARGET_VOCAB).exists():
            msg = "target vocab file %s is not found" % args.TARGET_VOCAB
            raise FileNotFoundError(msg)
        if not Path(args.validation_source).exists():
            msg = "validation source datast file %s is not found" % args.validation_source
            raise FileNotFoundError(msg)
        if not Path(args.validation_target).exists():
            msg = "validation target datast file %s is not found" % args.validation_target
            raise FileNotFoundError(msg)
        if args.batchsize < 1:
            msg = "--batchsize has to be >= 1"
            ValueError(msg)
        if args.epoch < 1:
            msg = "--epoch has to be >= 1"
            ValueError(msg)
        if args.unit < 1:
            msg = "--unit has to be >= 1"
            ValueError(msg)
        if args.layer < 1:
            msg = "--layer has to be >= 1"
            ValueError(msg)
        if args.source_min_token < 1:
            msg = "--source_min_token has to be >= 1"
            ValueError(msg)
        if args.source_max_token <= args.source_min_token:
            msg = "--source_max_token has to be >= --source_min_token"
            ValueError(msg)
        if args.target_min_token < 1:
            msg = "--target_min_token has to be >= 1"
            ValueError(msg)
        if args.target_max_token <= args.target_min_token:
            msg = "--taget_max_token has to be >= --target_min_token"
            ValueError(msg)
        if args.log_interval < 1:
            msg = "--log_interval has to be >= 1"
            ValueError(msg)
        if args.validation_interval < 1:
            msg = "--validation_interval has to be >= 1"
            ValueError(msg)
    except Exception as ex:
        parser.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    for (key, value) in vars(args).items():
        print("%s : %s" % (key, value))

    return args


def seq2seq_pad_convert(batch, device):

    xs, ys = zip(*batch)

    pad_xs = convert.concat_examples(xs, device, padding=-1)
    pad_ys = convert.concat_examples(ys, device, padding=-1)

    return {'xs': pad_xs, 'ys': pad_ys}

def main():

    args = get_arguments()

    train_data = Seq2SeqDatasetBase(
        args.SOURCE,
        args.TARGET,
        args.SOURCE_VOCAB,
        args.TARGET_VOCAB,
        args.source_min_token,
        args.source_max_token,
        args.target_min_token,
        args.target_max_token
    )

    # print dataset configurations
    dataset_configurations = train_data.get_configurations
    for key, value in dataset_configurations.items():
        print(key + '\t' + str(value))

    # make configuration file and save them.
    record_settings(args.out, args, dataset_configurations)

    # setup model
    model = seq2seq(
        args.layer,
        len(train_data.get_source_word_ids),
        len(train_data.get_target_word_ids),
        args.unit,
        dropout_ratio=0.1
    )

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    if args.mode == 'reuse':
        serializers.load_npz(args.resume, model)

    # setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # setup iterator
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)

    # setup updater and trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'main/loss', 'main/prep',
             'validation/main/loss', 'validation/main/prep',
             'validation/main/bleu', ' elapsed_time']
        ),
        trigger=(args.log_interval, 'epoch')
    )
    trainer.extend(
        extensions.LogReport(
            ['epoch', 'main/loss', 'main/prep',
             'validation/main/loss', 'validation/main/prep',
             'validation/main/bleu', 'elapsed_time']
        ),
        trigger=(args.log_interval, 'epoch')
    )

    if args.slack:
        trainer.extend(
            SlackNortifier(
                ['epoch', 'main/loss', 'main/prep',
                 'validation/main/loss', 'validation/main/prep',
                 'validation/main/bleu', 'elapsed_time'],
                SLACK_URL,
                username=args.out,
                channel=SLACK_REPORT_CHANNEL_NAME
            ),
            trigger=(args.log_interval, 'epoch')
        )
    trainer.extend(extensions.ProgressBar())
    trainer.extend(
        extensions.snapshot(
            filename='snapshot_iter_{.updater.iteration}'
        ),
        trigger=(args.log_interval, 'epoch')
    )

    trainer.extend(
        extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            x_key='epoch',
            trigger=(args.validation_interval, 'epoch'),
            file_name='loss.png'
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/prep', 'validation/main/prep'],
            x_key='epoch',
            trigger=(args.log_interval, 'epoch'),
            file_name='prep.png'
        )
    )

    if args.validation_source and args.validation_target:
        test_data = Seq2SeqDatasetBase(
            args.validation_source,
            args.validation_target,
            args.SOURCE_VOCAB,
            args.TARGET_VOCAB,
            args.source_min_token,
            args.source_max_token,
            args.target_min_token,
            args.target_max_token,
        )

        test_iter = chainer.iterators.SerialIterator(test_data,
                                                     args.batchsize,
                                                     repeat=False,
                                                     shuffle=False)

        @chainer.training.make_extension()
        def translate(trainer):
            source, target = test_data[np.random.choice(len(test_data))]
            result = model.translate([model.xp.array(source)])[0]

            source_sentence = ' '.join(test_data.source_index2token(source)[1:-1])
            target_sentence = ' '.join(test_data.target_index2token(target)[1:-1])
            result_sentence = ' '.join(test_data.target_index2token(result))

            text = 'epoch: ' + str(trainer.updater.epoch) + '\n' + \
                'source sentence: ' + source_sentence + '\n' + \
                'target sentence: ' + target_sentence + '\n' + \
                'result sentence: ' + result_sentence

            post2slack(
                text=text,
                username=args.out,
                channel=SLACK_TRANSLATION_CHANNEL_NAME,
                slack_url=SLACK_URL
            )

        trainer.extend(
            extensions.Evaluator(
                test_iter,
                model,
                converter=convert,
                device=args.gpu
            ),
            trigger=(args.validation_interval, 'epoch')
        )

        if args.slack:
            trainer.extend(
                translate,
                trigger=(args.validation_interval, 'epoch')
            )

        trainer.extend(
            CalculateBleu(
                model,
                test_data,
                'validation/main/bleu',
                batch=args.batchsize,
                device=args.gpu,
                n_grams=4
            ),
            trigger=(args.validation_interval, 'epoch')
        )

    print('start training')
    trainer.run()

    serializers.save_npz(Path(args.out) / 'model_final', model)
    serializers.save_npz(Path(args.out) / 'optimizer_final', optimizer)


if __name__ == '__main__':
    main()
