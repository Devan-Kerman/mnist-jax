import os
import time

os.environ['JAX_DEBUG_NANS'] = 'True'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import jax
import jax.numpy as jnp  # JAX NumPy

print(jax.devices())

from src.model import CNN

print("Initializing Model...")
cnn = CNN()
print(cnn.tabulate(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1))))

num_epochs = 10
batch_size = 32

print("Loading Dataset...")
from src.load import get_datasets, prefetch_iterable

train_ds, test_ds = get_datasets(num_epochs, batch_size)

print("Creating Training State...")
from src.train import create_train_state, compute_metrics, train_step

init_rng = jax.random.PRNGKey(0)
learning_rate = 0.01
momentum = 0.9
state = create_train_state(cnn, init_rng, learning_rate, momentum)
del init_rng  # Must not be used anymore.

metrics_history = {'train_loss': [],
				   'train_accuracy': [],
				   'test_loss': [],
				   'test_accuracy': []}

print("Starting Training!")
for epoch, (train_batches, test_batches) in enumerate(zip(train_ds, test_ds)):
	i = 0
	total_fetch = 0
	fetch = start = time.time()
	for batch in train_batches:
		total_fetch += time.time() - fetch
		state = train_step(state, batch)  # get updated train state (which contains the updated parameters)
		state = compute_metrics(state=state, batch=batch)  # aggregate batch metrics
		fetch = time.time()
		i += 1
	end = time.time()

	print(f"Total prefetch wait time: {total_fetch * 1000:5.2f}ms for {i} entries!")
	print(f"Trained epoch: {epoch + 1} in {end - start:5.2f}s!")
	for metric, value in state.metrics.compute().items():  # compute metrics
		metrics_history[f'train_{metric}'].append(value)  # record metrics

	state = state.replace(metrics=state.metrics.empty())  # reset train_metrics for next training epoch
	print(f"Train epoch:   {epoch + 1}, "
		  f"loss: {metrics_history['train_loss'][-1]}, "
		  f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")

	# Compute metrics on the test set after each training epoch
	start = time.time()
	test_state = state
	for test_batch in test_batches:
		test_state = compute_metrics(state=test_state, batch=test_batch)
	end = time.time()

	print(f"Tested epoch: {epoch + 1} in {end - start:5.2f}s!")
	for metric, value in test_state.metrics.compute().items():
		metrics_history[f'test_{metric}'].append(value)
	print(f"Test epoch:   {epoch + 1}, "
		  f"loss: {metrics_history['test_loss'][-1]}, "
		  f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")

import matplotlib.pyplot as plt  # Visualization

# Plot loss and accuracy in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Loss')
ax2.set_title('Accuracy')
for dataset in ('train', 'test'):
	ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
	ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
ax1.legend()
ax2.legend()
plt.show()
plt.clf()
