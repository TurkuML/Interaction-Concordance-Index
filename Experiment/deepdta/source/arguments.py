import argparse
import os

def argparser():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--seq_window_lengths',
      type=int,
      nargs='+',
      help='Space seperated list of motif filter lengths. (ex, --seq_window_lengths 4 8 12)'
  )
  parser.add_argument(
      '--smi_window_lengths',
      type=int,
      nargs='+',
      help='Space seperated list of motif filter lengths. (ex, --smi_window_lengths 4 8 12)'
  )
  parser.add_argument(
      '--num_windows',
      type=int,
      nargs='+',
      help='Space seperated list of the number of motif filters corresponding to length list. (ex, --num_windows 100 200 100)'
  )
  parser.add_argument(
      '--max_seq_len',
      type=int,
      default=0,
      help='Length of input sequences.'
  )
  parser.add_argument(
      '--max_smi_len',
      type=int,
      default=0,
      help='Length of input sequences.'
  )
  parser.add_argument(
      '--num_epoch',
      type=int,
      default=100,
      help='Number of epochs to train.'
  )
  #parser.add_argument( 
  #    '--num_hidden',
  #    type=int,
  #    default=0,
  #    help='Number of neurons in hidden layer.'
  #)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=256,
      help='Batch size. Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--dataset_path',
      type=str,
      default='data/kiba/',
      help='Directory for input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='tmp/',
      help='Directory for log data.'
  )
  parser.add_argument(
      '--is_log',
      action='store_true',
      help='Use log transformation for Y'
  )
  parser.set_defaults(is_log=False)

  parser.add_argument(
      '--learning_rates',
      type=float,
      nargs='+',
      default=[0.001],
      help='Space separated list of learning rates to try (e.g. 0.01, 0.001)'
  )

  parser.add_argument(
          '--fourfield_setting',
          type=str,
          help='Data split setting to use when running in fourfield mode. Should be S1, S2, S3 or S4.'
  )

  parser.add_argument(
          '--crossvalidation',
          action='store_true',
          help='Use cross validation, save predictions to .csv'
  )
  parser.set_defaults(crossvalidation=False)

  parser.add_argument(
          '--cv_fold',
          type=int,
          default=0,
          help='Which fold to train on and predict for when doing crossvalidation.'
  )

  parser.add_argument(
          '--cv_filename',
          type=str,
          default='',
          help='Pre-made splits.csv file to use when running in crossvalidation mode. Appended to --dataset_path.'
  )
  parser.add_argument(
          '--metricsearch',
          action='store_true',
          help='Search hyperparameters according to different metrics'
  )
  parser.set_defaults(metricsearch=False)

  parser.add_argument(
          '--save_freq',
          type=int,
          default=1,
          help='When using SavePredictionsCallback, save predictions every save_freq epochs'
  )

  FLAGS, unparsed = parser.parse_known_args()
  return FLAGS


def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)
  print("Logging:",msg)
