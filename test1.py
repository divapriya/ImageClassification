import os
import numpy as np
import tensorflow as tf

from scipy import ndimage


image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
num_labels=0

def load_letter():
  global num_labels
  """Load the data for a single letter label."""
  #image_files = os.listdir(folder)
  dataset_names = []
  dataset = np.ndarray(shape=(20000, image_size, image_size),
                             dtype=np.float32)
  dataset_label=np.ndarray(shape=(20000),dtype=object)
  num_images = 0
  flag=0
  data_folders=os.listdir('/Users/divalicious/Desktop/notMNIST_small')
  for folder in data_folders[1:]:
      print folder
      num_labels=num_labels+1
      # if flag==1:
      #   break
      letter=folder
      folder="/Users/divalicious/Desktop/notMNIST_small/"+folder
      image_files = os.listdir(folder)
      
      for image in image_files:
        image_file = os.path.join(folder, image)
        try:
          image_data = (ndimage.imread(image_file).astype(float) - 
                        pixel_depth / 2) / pixel_depth
          if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
          dataset[num_images, :, :] = image_data
          dataset_label[num_images]=letter
          num_images = num_images + 1
          # if num_images>199:
          #   flag=1
          #   break
        except IOError as e:
          print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
        
      # dataset = dataset[0:num_images, :, :]
      # if num_images < min_num_images:
      #   raise Exception('Many fewer images than expected: %d < %d' %
      #                   (num_images, min_num_images))
        
      print('Full dataset tensor:', dataset.shape)
      print('Mean:', np.mean(dataset))
      print('Standard deviation:', np.std(dataset))
  return dataset,dataset_label


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

dataset,dataset_label = load_letter()
dataset,dataset_label=randomize(dataset,dataset_label)
print dataset,dataset_label


train_dataset=dataset[:15000,:,]
train_labels = dataset_label[:15000]
valid_dataset =dataset[15000:18000,:,]
valid_labels = dataset_label[15000:18000]
test_dataset=dataset[18000:,:,] 
test_labels=dataset_label[18000:]
print test_labels
#print(np.argmax(test_labels,1))


def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print "reformattted"

train_subset = 10000
graph = tf.Graph()
with graph.as_default():
  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  tf_train_labels = tf.constant(train_labels[:train_subset])
  tf_test_labels = tf.constant(test_labels[train_subset:])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
  #prediction=tf.argmax(tf.nn.softmax(tf.matmul(X_placeholder, weights) + biases),1)

num_steps = 800

def accuracy(predictions, labels):
  print labels
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  test_labels
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  #tf.global_variables_initializer().run()
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:train_subset, :]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  print(np.argmax(test_prediction.eval(),1))
  #print(np.argmax(tf_test_labels,1))
  #print(tf.run(test_prediction,tf_test_dataset))
