# -*- encoding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description='Pipeline commandline argument')

# parameters for dataset settings
parser.add_argument("--train_file", type=str, default='', help="train_file")
parser.add_argument("--valid_file", type=str, default='', help="valid_file")
parser.add_argument("--model_path", type=str, default='', help="model_path")

parser.add_argument("--input_dim", type=int, default='22', help="input_dim")
parser.add_argument("--hidden_dim", type=int, default='128', help="hidden_dim")
parser.add_argument("--num_layers", type=int, default='5', help="num_layers")
parser.add_argument("--dropout", type=float, default='0.2', help="dropout")
parser.add_argument("--max_seq_len", type=int, default='192', help="max_seq_len")

parser.add_argument("--log_step", type=int, default='10', help="log_step")
parser.add_argument("--eval_step", type=int, default='50', help="eval_step")
parser.add_argument("--save_step", type=int, default='50', help="save_step")

config = parser.parse_args()