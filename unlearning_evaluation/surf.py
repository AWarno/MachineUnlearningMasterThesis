# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""The FairFace dataset."""
import glob
import os
import random

from absl import logging
import numpy as np
import sklearn
from sklearn import model_selection
import torch
from torch.utils import data
from torchvision import io
import torchvision.transforms as T
import datasets
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torchvision import transforms



# Function to convert bytes to PIL images
def bytes_to_pil(example):
    image = Image.open(BytesIO(example['img_bytes']))  # Convert bytes to PIL image
    example['img'] = image
    return example

# Function to resize images
def resize_image(example, size=(32, 32)):
    image = example['img']
    image = image.resize(size)  # Resize image to the desired size
    example['img'] = image
    return example


class FairFaceDataset(data.Dataset):
  """The FairFace dataset."""

  def __init__(self, path, width=224, height=224, split='train', limit: int = 10000):
    super().__init__()
    self.path = path

    # Load the dataset
    ds = datasets.load_dataset('nateraw/fairface', trust_remote_code=True)

    # Limit the dataset to the first 10 examples
    ds = ds['train'].select(range(limit))  # Use select to limit the number of examples

    # Apply the conversion to PIL images
    ds = ds.map(bytes_to_pil)

    # Apply the resize transformation
    ds = ds.map(lambda x: resize_image(x, size=(width, height)))


    self.examples = [{'image': sample['img'], 'age_group': sample['age'], 'age': sample['age']} for sample in ds]
    self.transform = ToTensor() 
    # Now, shuffle the examples, with the same random seed each time,
    # to ensure the same train / validation and test splits each time.
    random.Random(43).shuffle(self.examples)

    num_examples = len(self.examples)
    if split == 'train':
      self.examples = self.examples[: int(0.8 * num_examples)]
    elif split == 'val':
      self.examples = self.examples[
          int(0.8 * num_examples) : int(0.9 * num_examples)
      ]
    elif split == 'test':
      self.examples = self.examples[int(0.9 * num_examples) :]
    else:
      raise ValueError('Unknown split {}'.format(split))

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx):
    example = self.examples[idx].copy()
    image = example['image']
    image = transforms.ToTensor()(example['image'])
    example['image'] = image
    return example


def _get_age_group_counts(ds, name, quiet=False):
  """Get the age group counts."""
  age_group_counts = {}
  age_group_ranges = {}
  for sample in ds:
    age_group = sample['age_group']
    age = sample['age']

    if age_group in age_group_counts:
      age_group_counts[age_group][0] += 1
      age_group_ranges[age_group][0] = min(age_group_ranges[age_group][0], age)
      age_group_ranges[age_group][1] = max(age_group_ranges[age_group][1], age)
    else:
      age_group_counts[age_group] = [1, age]
      age_group_ranges[age_group] = [age, age]

  sorted_counts = sorted(age_group_counts.items(), key=lambda x: x[1][1])
  for age_group, (count, age) in sorted_counts:
    if not quiet:
      logging.info(
          '[Dataset %s] Age group %s : %d',
          name,
          age_group,
          round(100 * count / len(ds), 2),
      )
  return sorted_counts


def get_dataset(batch_size=64, quiet=False, dataset_path=''):
  """Get the Fairface dataset."""
  train_ds = FairFaceDataset(dataset_path, width=64, height=64, split='train')
  val_ds = FairFaceDataset(dataset_path, width=64, height=64, split='val')
  test_ds = FairFaceDataset(dataset_path, width=64, height=64, split='test')

  forget_size = int(0.1 * len(train_ds))  # 10% for forget
  retain_size = len(train_ds) - forget_size
  
  train_indices = np.arange(len(train_ds))
  retain_indices, forget_indices = train_test_split(train_indices, test_size=forget_size, random_state=0, shuffle=False)
  
  retain_ds = data.Subset(train_ds, retain_indices)
  forget_ds = data.Subset(train_ds, forget_indices)

  if not quiet:
    logging.info('Train set size %d', len(train_ds))
    logging.info('Val set size %d', len(val_ds))
    logging.info('Test set size: %d', len(test_ds))
    logging.info('Retain set size: %d', len(retain_ds))
    logging.info('Forget set size: %d', len(forget_ds))

  train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_loader = data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
  test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
  retain_loader = data.DataLoader(
      retain_ds, batch_size=batch_size, shuffle=True
  )
  forget_loader = data.DataLoader(
      forget_ds, batch_size=batch_size, shuffle=True
  )
  forget_loader_no_shuffle = data.DataLoader(
      forget_ds, batch_size=batch_size, shuffle=False
  )

  # Get the class weights:
  sorted_counts = _get_age_group_counts(train_ds, 'train', quiet=quiet)
  _get_age_group_counts(retain_ds, 'retain', quiet=quiet)
  _get_age_group_counts(forget_ds, 'forget', quiet=quiet)
  _get_age_group_counts(val_ds, 'valid', quiet=quiet)
  _get_age_group_counts(test_ds, 'test', quiet=quiet)
  class_weights = [
      1.0 / item[1][0] if item[1][0] != 0 else 1.0 for item in sorted_counts
  ]
  class_weights_tensor = torch.FloatTensor(class_weights)
  return (
      train_loader,
      val_loader,
      test_loader,
      retain_loader,
      forget_loader,
      forget_loader_no_shuffle,
      class_weights_tensor,
  )


def compute_accuracy_surf(
    data_names_list,
    data_loader_list,
    net,
    model_name,
    print_per_class_=True,
    print_=True,
):
  """Compute the accuracy."""
  net.eval()
  accs = {}
  pc_accs = {}
  list_of_classes = list(range(10))
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  with torch.no_grad():
    for name, loader in zip(data_names_list, data_loader_list):
      correct = 0
      total = 0
      correct_pc = [0 for _ in list_of_classes]
      total_pc = [0 for _ in list_of_classes]
      for sample in loader:
        inputs = sample['image']
        targets = sample['age_group']
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        for c in list_of_classes:
          num_class_c = (targets == c).sum().item()
          correct_class_c = (
              ((predicted == targets) * (targets == c)).float().sum().item()
          )
          total_pc[c] += num_class_c
          correct_pc[c] += correct_class_c

      accs[name] = 100.0 * correct / total
      pc_accs[name] = [
          100.0 * c / t if t > 0 else -1.0 for c, t in zip(correct_pc, total_pc)
      ]

  print_str = '%s accuracy: ' % model_name
  for name in data_names_list:
    print_str += '%s: %.2f, ' % (name, accs[name])
  if print_:
    logging.info(print_str)
  if print_per_class_:
    for name in data_names_list:
      print_str = '%s accuracy per class: ' % name
      for _, pc_acc in enumerate(pc_accs[name]):
        print_str += ' %.2f, ' % pc_acc
      logging.info(print_str)
  return accs
