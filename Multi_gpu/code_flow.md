# Tensorflow multi-gpu 사용법

### 1. Mirrored Strategy

- tf.distribute.MirroredStrategy()를 사용하여 strategy 생성.

- args가 비어었으면 준비된 모든 GPU 활성화. 만약 특정 GPU를 활성화시키고 싶다면 다음과 같이 사용.

  ```python
  strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
  ```

  

### 2. 데이터셋 전처리

- 만들어진 strategy에 대해서 distribute dataset 생성. -> 분할 연산에 적합한 형태로 입력 데이터 크기 변경.

  ```python
  def input_fn(BUFFER_SIZE, BATCH_SIZE):
      fashion_mnist = tf.keras.datasets.fashion_mnist
      (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
      train_images = train_images[..., None]
      test_images = test_images[..., None]
  
      train_images = train_images / np.float32(255)
      test_images = test_images / np.float32(255)
      train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(
          BATCH_SIZE)
      train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
  
      test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)
      test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
  
      return train_dist_dataset, test_dist_dataset
  ```



### 3. Distribute train step 생성

- 배치별 입력 데이터로 train step 생성.

- with strategy.scope()를 사용하면 연산이 자동으로 분할되어 각각의 GPU에 할당됨.

- strategy.scope() 내부에서 with tf.GradientTape()을 사용하여 gradient 연산 및 업데이트(step_fn)

- gradient연산이 끝나면 strategy.run(step_fn, args=(...))으로 위에서 만든 flow 실행.

  > 2.3버전까지는 tf.distrbute.Strategy.experimental_run_v2()를 사용했지만 2.4버전부터 experimental이 붙은 많은 부분을 정식지원. args는 같은 형태를 띔

- strategy.reduce()로 나누었던 연산 합치기. 합칠때 방식에 따라 옵션 선택 가능

  - tf.distribute.ReduceOp.SUM : 모두 더해서 합침
  - tf.distribute.ReduceOp.MEAN : 평균 내어 합침

- 분산 연산 종료

  ```python
  @tf.function
  def distributed_train_step(dataset_inputs, model, optimizer):
      image, label = dataset_inputs
  
      def step_fn(image, label):
          with tf.GradientTape() as tape:
              pred = model(image)
              grtr = label
              loss = get_loss(pred, grtr) / BATCH_SIZE
  
          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(grads, model.trainable_variables))
          return loss[tf.newaxis]
  
      per_example_losses = strategy.run(step_fn, args=(image, label))
      strategy.reduce(tf.distribute.ReduceOp.)
      sum_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
      return sum_loss
  ```

  