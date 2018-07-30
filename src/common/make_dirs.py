from pathlib import Path

def create_save_dirs(args):

    result_dir = str(args.batchsize) + '-' + \
                 str(args.epoch) + '-Adam-' + \
                 str(args.layer) + '-' + \
                 str(args.unit)

    out_dir = Path(args.out)
    base_dir = out_dir / result_dir

    log_dir = base_dir / 'logs'
    plot_dir = base_dir / 'plots'
    other_dir = base_dir / 'others'
    result_dir = base_dir / 'models'
    snapshot_dir = result_dir / 'snapshot'
    snapshot_trainer_dir = snapshot_dir / 'trainer'
    snapshot_model_dir = snapshot_dir / 'models'
    snapshot_opt_dir = snapshot_dir / 'optimizers'
    final_result = result_dir / 'final_result'

    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    if not base_dir.exists():
        base_dir.mkdir()
    if not log_dir.exists():
        log_dir.mkdir()
    if not plot_dir.exists():
        plot_dir.mkdir()
    if not other_dir.exists():
        other_dir.mkdir()
    if not result_dir.exists():
        result_dir.mkdir()
    if not snapshot_dir.exists():
        snapshot_dir.mkdir()
    if not snapshot_trainer_dir.exists():
        snapshot_trainer_dir.mkdir()
    if not snapshot_model_dir.exists():
        snapshot_model_dir.mkdir()
    if not snapshot_opt_dir.exists():
        snapshot_opt_dir.mkdir()
    if not final_result.exists():
        final_result.mkdir()

    result = {'base_dir':base_dir,
              'log_dir': log_dir,
              'plot_dir': plot_dir,
              'snapshot_dir': snapshot_dir,
              'snapshot_trainer_dir': snapshot_trainer_dir,
              'snapshot_model_dir': snapshot_model_dir,
              'snapshot_opt_dir': snapshot_opt_dir,
              'final_result': final_result}

    return result
