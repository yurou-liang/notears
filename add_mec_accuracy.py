"""Add CPDAG accuracy metrics to saved experiment JSON files in place.

Examples (run with the project's Python environment):
    python add_mec_accuracy.py linear_exist_trek_pairs_soft
    python add_mec_accuracy.py linear_exist_trek_pairs_soft --keys W_est_prior_ll
    python add_mec_accuracy.py linear_exist_trek_pairs_soft --keys W_est_prior_ll W_est_prior_l2

By default all four prior/no-prior likelihood/L2 cases are evaluated. Missing
cases are skipped. Invalid matrices are reported without preventing other cases
from being processed. Undefined F1 values are saved as JSON null.
"""
import argparse
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile


DEFAULT_KEYS = (
    'W_est_prior_ll', 'W_est_prior_l2',
    'W_est_no_prior_ll', 'W_est_no_prior_l2',
)


def atomic_write(path, result):
    """Replace a file only after its complete updated JSON has been written."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def update_file(path, keys, evaluator, dry_run=False):
    import numpy as np

    with path.open(encoding='utf-8') as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object')
    if 'B_true' not in result:
        raise ValueError('Missing B_true')
    truth = np.asarray(result['B_true'])
    updated, missing, failed = [], [], []
    for key in keys:
        if key not in result:
            missing.append(key)
            continue
        try:
            weights = np.asarray(result[key], dtype=float)
            if not np.isfinite(weights).all():
                raise ValueError('Weights contain NaN or infinity')
            scores = evaluator(truth, weights != 0)
            # The selected function returns NaN for undefined F1 scores.
            scores = {
                name: None if isinstance(value, float) and not math.isfinite(value) else value
                for name, value in scores.items()
            }
            output_key = 'acc_MEC_' + key.removeprefix('W_est_')
            result[output_key] = scores
            updated.append(output_key)
        except Exception as exc:
            failed.append((key, str(exc)))
    if updated and not dry_run:
        atomic_write(path, result)
    return updated, missing, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', type=Path, help='Folder containing experiment JSON files')
    parser.add_argument('--keys', nargs='+', default=list(DEFAULT_KEYS), metavar='W_EST_KEY',
                        help='W_est_* fields to evaluate (default: all four cases)')
    parser.add_argument('--dry-run', action='store_true', help='Compute and report metrics without writing files')
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error(f'Folder does not exist: {args.folder}')
    if any(not key.startswith('W_est_') or not key.removeprefix('W_est_') for key in args.keys):
        parser.error('Each matrix key must start with W_est_ and include a case suffix')
    paths = sorted(args.folder.glob('*.json'))
    if not paths:
        parser.error(f'No JSON files found in {args.folder}')
    keys = list(dict.fromkeys(args.keys))
    try:
        from prior_notears.utils import count_MEC_accuracy
    except ModuleNotFoundError as exc:
        parser.error(
            f'Cannot import count_MEC_accuracy: missing {exc.name!r}. '
            'Use a project environment with its dependencies, including causal-learn.'
        )

    files_updated = metrics_updated = missing_count = failed_count = 0
    for path in paths:
        try:
            updated, missing, failed = update_file(path, keys, count_MEC_accuracy, args.dry_run)
        except Exception as exc:
            print(f'ERROR {path}: {exc}', file=sys.stderr)
            failed_count += 1
            continue
        if updated:
            files_updated += 1
            metrics_updated += len(updated)
            print(f'{path.name}: {", ".join(updated)}')
        missing_count += len(missing)
        for key in missing:
            print(f'SKIP {path.name}: missing {key}', file=sys.stderr)
        for key, error in failed:
            print(f'ERROR {path.name} [{key}]: {error}', file=sys.stderr)
            failed_count += 1
    action = 'Would update' if args.dry_run else 'Updated'
    print(f'{action} {metrics_updated} metric entries in {files_updated}/{len(paths)} files; '
          f'{missing_count} missing cases skipped; {failed_count} errors.')
    return 1 if failed_count else 0


if __name__ == '__main__':
    raise SystemExit(main())
