import collections
import os
import jax
import numpy as np
from datasets import load_dataset
import itertools
import time
import joblib

threads = os.cpu_count() or 1


def prefetch_iterable(values, size=2):
	queue = collections.deque()

	iterator = iter(values)

	def enqueue(n):
		for value in itertools.islice(iterator, n):
			queue.append([jax.device_put(value)])

	enqueue(size)
	while queue:
		ret = queue.popleft()[0]
		enqueue(1)
		yield ret
		jax.device_get(ret)


def normalize(entry):
	return {'label': entry['label'], 'image': entry['image'].reshape(28, 28, 1) / 255.0}


def make_batch(entries):
	copy = {}
	for k, v in entries.items():
		copy[k] = [np.array(v)]
	return copy


def map_iter(collect, map_func):
	return joblib.Parallel(n_jobs=min(len(collect), threads))(joblib.delayed(map_func)(i) for i in collect)


def get_datasets(num_epochs, batch_size):
	"""Load MNIST train and test datasets into memory."""
	ds = load_dataset("mnist")
	ds = ds.with_format("numpy")

	train_ds = ds["train"]
	test_ds = ds["test"]

	train_ds = train_ds.map(normalize, num_proc=threads).shuffle(1024)
	test_ds = test_ds.map(normalize, num_proc=threads)

	train_ds = train_ds.map(make_batch, num_proc=threads, batched=True, batch_size=batch_size)
	test_ds = test_ds.map(make_batch, num_proc=threads, batched=True, batch_size=batch_size)

	print(f"{train_ds.shape}")

	# train_ds = [train_ds.shard(num_shards=num_epochs, index=i) for i in range(num_epochs)]
	# test_ds = [test_ds.shard(num_shards=num_epochs, index=i) for i in range(num_epochs)]

	start = time.time()
	train_ds = map_iter([*range(num_epochs)], lambda i: list(train_ds.shard(num_shards=num_epochs, index=i)))
	test_ds = map_iter([*range(num_epochs)], lambda i: list(test_ds.shard(num_shards=num_epochs, index=i)))
	print(f"{time.time() - start:5.2f}s to convert dataset to list")

	train_ds = prefetch_iterable(train_ds)
	test_ds = prefetch_iterable(test_ds)

	return train_ds, test_ds
