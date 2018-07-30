from nltk.translate import bleu_score
import chainer

class CalculateBleu(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self,
            model,
            test_data,
            key,
            batch=100,
            device=-1,
            max_length=100,
            n_grams=4
    ):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length
        self.n_grams = n_grams
        self.weights = [1./n_grams for _ in range(0, n_grams)]

    def __call__(self, iterator):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist() for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references,
            hypotheses,
            self.weights,
            smoothing_function=bleu_score.SmoothingFunction().method1
        )
        chainer.report({self.key: bleu})
