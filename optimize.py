"""Find the detector design that gives the highest Space Py Quest score.

The score maximised is the "Total Range" reported by the Science Run button
(ScoreCalculator.CalcScore().score). A design is only accepted if

  * its cost is within the budget of the chosen site, and
  * its complexity is within the site's complexity credits, so no detections
    are missed (disable with --allow-missed).

Every site/material combination is searched with several independent
differential evolution runs, followed by a hill climb on the slider grid.
Continuous parameters are snapped to the 0.1 step of the notebook sliders and
pumps / suspension stages are whole numbers, so the result can be typed
straight into the game.

Usage examples:
   python optimize.py                      # search everything
   python optimize.py --site Desert        # only one site
   python optimize.py --material Silicon --maxiter 300 --seed 1
"""
import argparse
import multiprocessing
import warnings

import numpy as np
from scipy.optimize import differential_evolution

import pystq.materials as materials
import pystq.sites as sites
from pystq.detector import Detector
from pystq.score import CalcComplex, CalcCost, ScoreCalculator

SITES = ['City', 'Jungle', 'Desert', 'Island']
MATERIALS = ['Crystal', 'Silicon', 'Sapphire', 'Silica']

# (key, lower, upper, step). Bounds are the slider limits in detector.py.
PARAMS = [
   ('depth', 0.0, 1000.0, 0.1),
   ('pumps', 1, 16, 1),
   ('temperature', 1.0, 330.0, 0.1),
   ('sus_stages', 1, 9, 1),
   ('sus_length', 0.35, 5.0, 0.01),
   ('mirror_mass', 5.0, 100.0, 0.1),
   ('power', 1.0, 200.0, 0.1),
   ('roughness', 1, 500, 0.1),
]

# Infeasible designs score worse than any feasible one
PENALTY = 1e6


def snap(x):
   """Round a raw parameter vector onto the game's slider grid."""
   values = {}
   for xi, (key, lo, hi, step) in zip(x, PARAMS):
      v = round(round(xi / step) * step, 10)
      v = min(max(v, lo), hi)
      values[key] = int(v) if step == 1 else v
   return values


def build_detector(values, site, material, freqrange):
   params = dict(values)
   params['site'] = getattr(sites, site)
   params['material'] = getattr(materials, material)
   params['freqrange'] = freqrange
   return Detector(params)


def evaluate(values, site, material, freqrange):
   det = build_detector(values, site, material, freqrange)
   with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      score = ScoreCalculator(det).CalcScore()
   return det, score, CalcCost(det), CalcComplex(det)


def objective(x, site, material, freqrange, allow_missed):
   values = snap(x)
   det = build_detector(values, site, material, freqrange)
   site_cls = det.parameters['site']

   # Cheap constraint checks first; skip the noise integrals if infeasible
   violation = max(0.0, CalcCost(det) / site_cls.budget - 1)
   if not allow_missed:
      violation += max(0.0, CalcComplex(det) / site_cls.complexCredits - 1)
   if violation > 0:
      return PENALTY * (1 + violation)

   with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      score = ScoreCalculator(det).CalcScore().score
   if not np.isfinite(score):
      return PENALTY
   return -score


def refine(values, site, material, freqrange, allow_missed):
   """Greedy hill climb on the slider grid to squeeze out the last bits."""
   def f(v):
      x = [v[key] for key, *_ in PARAMS]
      return objective(x, site, material, freqrange, allow_missed)

   best = f(values)
   multipliers = [100, 10, 1]
   improved = True
   while improved:
      improved = False
      for key, lo, hi, step in PARAMS:
         for mult in multipliers:
            for sign in (-1, 1):
               trial = dict(values)
               trial[key] = values[key] + sign * mult * step
               if not lo <= trial[key] <= hi:
                  continue
               trial = snap([trial[k] for k, *_ in PARAMS])
               val = f(trial)
               if val < best:
                  best, values, improved = val, trial, True
   return values, best


def optimize_combo(args):
   site, material, opts = args
   fargs = (site, material, opts['freqrange'], opts['allow_missed'])
   best_values, best_fun = None, np.inf
   for restart in range(opts['restarts']):
      result = differential_evolution(
         objective,
         bounds=[(lo, hi) for _, lo, hi, _ in PARAMS],
         args=fargs,
         maxiter=opts['maxiter'],
         popsize=opts['popsize'],
         tol=1e-8,
         mutation=(0.5, 1.0),
         recombination=0.7,
         seed=opts['seed'] + restart,
         polish=False,
         init='sobol',
      )
      if result.fun < best_fun:
         best_values, best_fun = snap(result.x), result.fun

   if best_fun < PENALTY:
      best_values, best_fun = refine(best_values, *fargs)
   feasible = best_fun < PENALTY
   return site, material, best_values, (-best_fun if feasible else None)


def describe(site, material, values, freqrange):
   det, s, cost, complexity = evaluate(values, site, material, freqrange)
   site_cls = det.parameters['site']
   lines = [
      'Site: {}   Material: {}'.format(site, material),
      'Total Range (score): {:.2f} Mpc'.format(s.score),
      '  NSNS range: {:.2f} Mpc   ({:.0f} detections, {} missed)'.format(
         s.nsnsRange, s.nsns, s.nsnsMissed),
      '  BHBH range: {:.2f} Mpc   ({:.0f} detections, {} missed)'.format(
         s.bhbhRange, s.bhbh, s.bhbhMissed),
      '  Supernovae: {:.0f} ({} missed)'.format(s.supernovae,
                                                s.supernovaeMissed),
      'Cost: ${:,.0f} of ${:,.0f} budget (${:,.0f} remaining)'.format(
         cost, site_cls.budget, site_cls.budget - cost),
      'Complexity: {:.2f} of {} credits'.format(complexity,
                                                site_cls.complexCredits),
      '',
      'Paste into the notebook:',
      'best = {',
   ]
   for key, *_ in PARAMS:
      lines.append("    '{}' : {},".format(key, values[key]))
   lines.append("    'site' : sites.{},".format(site))
   lines.append("    'material' : materials.{},".format(material))
   lines.append('}')
   lines.append('stq = spaceTimeQuest(best)')
   return '\n'.join(lines)


def main():
   parser = argparse.ArgumentParser(
      description='Optimise Space Py Quest detector parameters.')
   parser.add_argument('--site', choices=SITES, nargs='+', default=SITES)
   parser.add_argument('--material', choices=MATERIALS, nargs='+',
                       default=MATERIALS)
   parser.add_argument('--maxiter', type=int, default=200,
                       help='differential evolution generations per combo')
   parser.add_argument('--popsize', type=int, default=20)
   parser.add_argument('--restarts', type=int, default=4,
                       help='independent searches per combo (more = safer)')
   parser.add_argument('--seed', type=int, default=0)
   parser.add_argument('--freqrange', type=float, nargs=2, default=(0, 4),
                       metavar=('LO', 'HI'),
                       help='log10 frequency range used for scoring '
                       '(the game default is 0 4)')
   parser.add_argument('--allow-missed', action='store_true',
                       help='allow complexity above the site credits '
                       '(detections will be missed)')
   parser.add_argument('--jobs', type=int, default=multiprocessing.cpu_count())
   args = parser.parse_args()

   opts = dict(maxiter=args.maxiter, popsize=args.popsize, seed=args.seed,
               restarts=args.restarts,
               freqrange=tuple(args.freqrange),
               allow_missed=args.allow_missed)
   combos = [(s, m, opts) for s in args.site for m in args.material]

   print('Optimising {} site/material combinations...'.format(len(combos)))
   with multiprocessing.Pool(args.jobs) as pool:
      results = []
      for site, material, values, score in pool.imap_unordered(
            optimize_combo, combos):
         label = '{:.2f} Mpc'.format(score) if score is not None \
            else 'no design fits budget/complexity'
         print('  {:<7} {:<9} {}'.format(site, material, label))
         results.append((site, material, values, score))

   feasible = [r for r in results if r[3] is not None]
   if not feasible:
      print('\nNo feasible design found.')
      return

   feasible.sort(key=lambda r: r[3], reverse=True)
   print('\nRanking:')
   for site, material, _, score in feasible:
      print('  {:8.2f} Mpc  {:<7} {}'.format(score, site, material))

   site, material, values, _ = feasible[0]
   print('\n===== Best design =====')
   print(describe(site, material, values, tuple(args.freqrange)))


if __name__ == '__main__':
   main()
