"""Wrapper script to score experiment results arranged as:

  experiments_root/
    <method>/
      <workload>/
        trial_<n>/
          eval_measurements.csv

This script validates the directory structure, collects per-method data
using `scoring.scoring_utils.get_experiment_df`, writes per-method summaries,
and optionally computes performance profiles + leaderboard scores.

Usage examples:
  python scoring/score_experiments.py \
    --experiments_root /scratch/global/chen8596/mlcommons_experiments \
    --output_dir scoring_results --compute_performance_profiles

"""
import argparse
import os
import pickle
from absl import logging

from scoring import scoring_utils
from scoring import score_submissions as ss
from scoring import performance_profile


def find_submission_dirs(experiments_root):
  return [
    d for d in os.listdir(experiments_root) if os.path.isdir(os.path.join(experiments_root, d))
  ]


def has_measurements(submission_path):
  # Heuristic: look for at least one file named eval_measurements.csv under
  # submission_path/<workload>/trial_*/eval_measurements.csv
  for workload in os.listdir(submission_path):
    wpath = os.path.join(submission_path, workload)
    if not os.path.isdir(wpath):
      continue
    for trial in os.listdir(wpath):
      tpath = os.path.join(wpath, trial)
      if not os.path.isdir(tpath):
        continue
      csv_path = os.path.join(tpath, scoring_utils.MEASUREMENTS_FILENAME)
      if os.path.exists(csv_path):
        return True
  return False


def main(argv=None):
  parser = argparse.ArgumentParser(description='Score experiments directory')
  parser.add_argument('--experiments_root', required=True)
  parser.add_argument('--output_dir', default='scoring_results')
  parser.add_argument('--compute_performance_profiles', action='store_true')
  parser.add_argument('--self_tuning_ruleset', action='store_true')
  parser.add_argument('--strict', action='store_true')
  parser.add_argument('--exclude_submissions', default='')
  parser.add_argument('--save_results_to_filename', default=None)
  parser.add_argument('--load_results_from_filename', default=None)
  parser.add_argument('--dry_run', action='store_true')
  args = parser.parse_args(argv)

  os.makedirs(args.output_dir, exist_ok=True)

  exclude = [e for e in args.exclude_submissions.split(',') if e]

  results = {}

  if args.load_results_from_filename:
    with open(os.path.join(args.output_dir, args.load_results_from_filename), 'rb') as f:
      results = pickle.load(f)
  else:
    methods = find_submission_dirs(args.experiments_root)
    logging.info(f'Found {len(methods)} candidate submissions in {args.experiments_root}')
    if args.dry_run:
      logging.info('Dry run: listing up to 20 submissions:')
      for m in methods[:20]:
        logging.info('  %s', m)

    for submission in methods:
      if submission in exclude:
        logging.info('Skipping excluded submission %s', submission)
        continue
      submission_path = os.path.join(args.experiments_root, submission)
      if not has_measurements(submission_path):
        logging.warning('No measurements found for %s; skipping', submission)
        continue
      logging.info('Reading submission %s', submission)
      df = scoring_utils.get_experiment_df(submission_path)
      if df is None or df.empty:
        logging.warning('Empty DataFrame for %s; skipping', submission)
        continue
      results[submission] = df

      # Per-submission summary (similar to score_submissions.py)
      try:
        summary_df = ss.get_submission_summary(df)
      except Exception:
        logging.exception('Failed to get submission summary for %s', submission)
        summary_df = None

      if summary_df is not None:
        summary_fn = os.path.join(args.output_dir, f'{submission}_summary.csv')
        summary_df.to_csv(summary_fn)
        logging.info('Saved summary %s', summary_fn)

    if args.save_results_to_filename:
      with open(os.path.join(args.output_dir, args.save_results_to_filename), 'wb') as f:
        pickle.dump(results, f)
        logging.info('Saved aggregated results to %s', args.save_results_to_filename)

  if args.compute_performance_profiles:
    if not results:
      raise ValueError('No results to compute performance profiles on.')
    perf_df = performance_profile.compute_performance_profiles(
      results,
      time_col='score',
      min_tau=1.0,
      max_tau=4.0,
      reference_submission_tag=None,
      num_points=100,
      scale='linear',
      verbosity=1,
      strict=args.strict,
      self_tuning_ruleset=args.self_tuning_ruleset,
      output_dir=args.output_dir,
    )

    performance_profile.plot_performance_profiles(
      perf_df, 'score', save_dir=args.output_dir
    )
    scores = performance_profile.compute_leaderboard_score(perf_df)
    scores.to_csv(os.path.join(args.output_dir, 'scores.csv'))
    logging.info('Saved performance profiles and scores to %s', args.output_dir)

  logging.info('Done.')


if __name__ == '__main__':
  main()
