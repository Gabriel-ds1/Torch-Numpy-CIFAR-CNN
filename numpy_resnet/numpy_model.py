import os
import pickle
import copy
from numpy_resnet.utils import backend
from common_utils.cifar10_data_loader import augment_batch
from numpy_resnet.model_builder.layers import Layer_Input
from numpy_resnet.model_builder.loss_functions import Activation_Softmax, Loss_CategoricalCrossentropy, Softmax_Loss_CatCrossentropy

# Model class
class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object (if used)
        self.softmax_classifier_output = None
    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # Set loss and optimizer
    # use of astrisk notes that the subsequent parameters (loss and optimizer in this case) are keyword arguments
    # since they have no default value assigned they are required to be passed by names and values, making the code more legible.
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    # Retrieves and returns parameters of trainable layers
    def get_parameters(self):
        # create a list for parameters
        parameters = []

        # iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters
    
    # Updates the model with new parameters
    def set_parameters(self, parameters):
        # iterate over the parameters and layers and update each layer with each set of parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # saves parameters to a file
    def save_parameters(self, path):
        # Open a file in the binary-write mode and save parameters
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    # Load the weights and updates a model instance with them
    def load_parameters(self, path):
        # open file in binary-read mode, load weights and update trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    # saves the model
    def save(self, path):
        # make a deep copy of current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # remove data from the input layer and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer remove inputs, outputs and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        
        # Open a file in binary-write mode and save model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    # loads and returns a model
    @staticmethod
    def load(path):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise ValueError(f"Model file '{path}' is missing or empty.")
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    # Predicts on the samples
    def predict(self, x, *, batch_size=None):
        # Default value if batch size is not being set
        prediction_steps = 1

        # calculate number of steps
        if batch_size is not None:
            prediction_steps = len(x) // batch_size
            # Dividing rounds down. If there are some remaining data, but not full batch, this wont include it
            # Add '1' to include this not full batch
            if prediction_steps * batch_size < len(x):
                prediction_steps += 1
        
        # model outputs
        output = []

        # Iterate over steps
        for step in range(prediction_steps):
            # if batch size is not set - train using one step and full dataset
            if batch_size is None:
                batch_x = x
            #otherwise slice a batch
            else:
                batch_x = x[step*batch_size:(step+1)*batch_size]
            # Perform a forward pass
            batch_output = self.forward(batch_x, training=False)

            # Append batch prediction to the list of predictions
            output.append(batch_output)
        # stack and return results
        return backend.np.vstack(output)

    # finalize the model
    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_Input()
        
        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):
            # If its the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # all layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            
            # the last layer - the next object is the loss, also let's save aside the reference to the last object whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains attribute called "weights" its a trainable layer - add to list of trainable layers
            # We dont need to check for biases, weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        #if output activation is softmax and loss function is categorical cross-entropy, create an object of combined activation and loss function
        # containing faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            # create an object of combined activation and loss functions
            self.softmax_classifier_output = Softmax_Loss_CatCrossentropy()

    # Train the model
    def train(self, x, y, *, epochs=1, batch_size=None, print_every=1, augment_data=False, validation_data=None):
        
        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not being set
        train_steps = 1

        # If there is validation data, set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1
            # for better readability
            x_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(x) // batch_size
            # dividing rounds down. if there are some remaining data, but not a full batch, this wont include it
            # add '1' to include this not full batch
            if train_steps * batch_size < len(x):
                train_steps += 1
            if validation_data is not None:
                validation_steps = len(x_val) // batch_size
                if validation_steps * batch_size < len(x_val):
                    validation_steps += 1

        # Main training loop
        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')

            # resest accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            # iterate over steps
            for step in range(train_steps):
                # if batch size is not set, train using one step and full dataset
                if batch_size is None:
                    batch_x = x
                    batch_y = y
                # otherwise slice a batch
                else:
                    batch_x = x[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                if augment_data:
                    # augment dataset with random crop, flip, rotation, color jitter and cutout
                    batch_x = augment_batch(batch_x)
                # perform the forward pass
                output = self.forward(batch_x, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # get predictions and calulcate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params(epoch)
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'step:{step}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' + f'data_loss: {data_loss:.3f}, ' + f'reg_loss: {regularization_loss:.3f}, ' + f'lr: {self.optimizer.current_learning_rate}')

        # get and print epoch loss and accuracy
        epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
        epoch_loss = epoch_data_loss + epoch_regularization_loss
        epoch_accuracy = self.accuracy.calculate_accumulated()
        print(f'training, ' + f'acc: {epoch_accuracy:.3f}, ' + f'loss: {epoch_loss:.3f} (' + f'data_loss: {epoch_data_loss:.3f}, ' + f'reg_loss: {epoch_regularization_loss:.3f}),' + f'lr: {self.optimizer.current_learning_rate}')

        if validation_data is not None:
            # evaluate the model
            self.evaluate(*validation_data, batch_size=batch_size)

    def evaluate(self, x_val, y_val, *, batch_size=None):
        # default value if batch size is not being set
        validation_steps = 1

        # calculate number of steps
        if batch_size is not None:
            validation_steps = len(x_val) // batch_size
            if validation_steps * batch_size < len(x_val):
                validation_steps += 1

        # Reset accumulated values in loss and accuracy objects
        #self.loss.new_pass()
        #self.accuracy.new_pass()

        # iterate over steps
        for step in range(validation_steps):
            # if batch size is not set, train using one step and full dataset
            if batch_size is None:
                batch_x = x_val
                batch_y = y_val
            #otherwise slice a batch
            else:
                batch_x = x_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            output = self.forward(batch_x, training=False)

            # calculate the loss
            self.loss.calculate(output, batch_y)

            # get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # print a summary
        print(f'validation, ' + f'acc: {validation_accuracy:.3f}, ' + f'loss: {validation_loss:.3f}')

    def forward(self, x, training):
        # Call the forward method on the input layer, this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(x, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as the parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list, return output
        return layer.output
    
    def backward(self, output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # first call backward method on the combined activation/loss. this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # since we'll not call backward method of the last layer which is softmax activation as we used combined activation/loss object, lets set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            #call backward method going through all the objects but last in reversed order passing dinputs as paramaeter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        # first call backward method on the loss, this will set dinputs property that the last layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects in reversed order, passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)