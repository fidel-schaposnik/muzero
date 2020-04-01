from games.tictactoe import *
from storage_replay import *
from evaluation import *
from training import *

config = make_tictactoe_config()
storage = SharedStorage(config)
# replay_buffer = ReplayBuffer(config)

network = storage.latest_network()
network.load_weights(r'checkpoints\TicTacToe\31-03-2020--20-08\TicTacToe_it9999')
# evaluate_agents(config, ['random', network], num_games=100)
# play_against_network(config, network, human_player_id=0)
#
# batch_selfplay(config, replay_buffer, network, num_games=1)
# print(replay_buffer.buffer[0])
#
# batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.discount)
# print(batch)
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
# synchroneous_train_network(config, storage, num_games=100, num_steps=100, num_eval_games=0, checkpoint_path='checkpoints')

game = config.new_game()
game.apply(Action(4))
state = np.expand_dims(game.make_image(),0)
print(state.shape)
hidden_state = network.representation(state)
print(network.prediction(hidden_state))
