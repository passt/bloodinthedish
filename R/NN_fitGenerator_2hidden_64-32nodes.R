
# Number of splits (copy from above)
ksplits = ksplits

# Initialize empty lists
modell = list()
modell.predict = list()
modell.predict.p = list()

history = list()

# Loop over k-folds
for (fold in 1:ksplits) {
  cat('Fold: ' ,fold, '\n')
  # extract indices for fold
  ix.test <- as.numeric(unlist(lapply(xv.kfold, function(x){x[[fold]]})))
  # extract training and test data for fold
  x.train <- x.Liver[-ix.test, ]
  x.test  <- x.Liver[ ix.test, ]
  
  # FUNCTIONAL API (DOES allow dropout in input)
  # Define the model
  inputs <- layer_input(shape = ncol(x.train))
  predictions <- inputs %>%
    # layer_dropout(rate = 0.5)                                %>% # Dropout
    layer_dense(units = 64, activation = 'relu',
                kernel_regularizer = regularizer_l1(l=0.001))  %>% # Dense (64)
    layer_dense(units = 32, activation = 'relu',
                kernel_regularizer = regularizer_l1(l=0.001))  %>% # Dense (32)
    layer_dense(units = 26, activation = 'softmax')              # Softmax (26)
  # Put pieces together
  modell[[fold]] <- keras_model(inputs = inputs, outputs = predictions)
  
  # Compile the model
  modell[[fold]] %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )
  
  # fit generator samples euqally across all classes (5 examples)
  gen <- function(){
    ix.batch <- tapply(1:nrow(x.train), y.Liver[-ix.test], sample, 16) %>% unlist %>% as.vector
    x.batch  <- x.train[ix.batch,]
    y.batch  <- y.train[-ix.test,][ix.batch,]
    return(list(x.batch, y.batch))
  }
  
  # Train the model
  history[[fold]] <- 
    modell[[fold]] %>% fit_generator(
      generator = gen, 
      steps_per_epoch = 256,
      epochs = 1000, 
      validation_data = list(x.test, y.train[ix.test,]),
      verbose = 0)
  
  # print val acc
  print(history[[fold]]$metrics %>% lapply(tail, 1))
  
  # predict probabilities for softmax classification
  modell.predict.p[[fold]] <- modell[[fold]] %>% predict(x.test)
  
  # Predict labels for validation data
  x.test.class <- modell.predict.p[[fold]] %>% apply(1, function(x){which.max(x)})
  modell.predict[[fold]] <-
    list( predicted = factor(x.test.class, labels = levels(y.Liver)),
          true = y.Liver[ix.test] )
  rm(x.test.class)
  
}