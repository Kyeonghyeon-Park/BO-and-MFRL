import argparse
from actor_critic_v3 import ActorCritic

designer_alpha = 0.8  # Bayesian optimization으로

parser = argparse.ArgumentParser()
parser.add_argument('--designer_alpha', default=designer_alpha)
parser.add_argument('--sample_size', default=4)
parser.add_argument('--buffer_max_size', default=50)
parser.add_argument('--max_episode_number', default=3500)
parser.add_argument('--discount_factor', default=1)
parser.add_argument('--epsilon', default=0.5)
parser.add_argument('--mean_action_sample_number', default=5)
parser.add_argument('--obj_weight', default=0.6)
parser.add_argument('--lr_actor', default=0.0001)
parser.add_argument('--lr_critic', default=0.001)
parser.add_argument('--update_period', default=10)
parser.add_argument('--trained', default=False)
parser.add_argument('--PATH', default='')
parser.add_argument('--filename', default='')

args = parser.parse_args()

#############
# if want to learn trained network more, try to do this code
# args.trained = True
# args.PATH = './weights/a_lr=0.0001_alpha=1/'
# args.filename = 'all.tar'
# args.max_episode_number = 3500
#############
# try other settings
# args.lr_actor = 0.0005
#############
model = ActorCritic(args)
model.run()