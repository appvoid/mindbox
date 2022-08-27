import mindbox.console as console
class Network:
    def __init__(self, debug):
        self.debug = debug
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate, counter, threshold):
        self.cancelled = False
        # sample dimension first
        self.counter = 0
        samples = len(x_train)
        # training loop
        epoch_animations = ['🕛','🕐','🕑','🕒','🕓','🕔','🕕','🕖','🕗','🕘','🕙','🕚']
        step_anim = 0
        from timeit import default_timer as timer 
        start = timer() # training time started
        for i in range(epochs):
            try:
                err = 0
                self.counter+=1
                # stops when a threshold is given
                if (threshold != None and threshold >= err): break
                
                for j in range(samples):
                    # forward propagation
                    output = x_train[j]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)

                    # compute loss (for display purpose only)
                    err += self.loss(y_train[j], output)

                    # backward propagation
                    error = self.loss_prime(y_train[j], output)
                    for layer in reversed(self.layers):
                        error = layer.backward_propagation(error, learning_rate)

                # calculate average error on all samples
                err /= samples
                self.error = err

                if (self.counter >= counter):
                    # handle simple animation for joyful feedback
                    try:
                        current_anim = epoch_animations[step_anim]
                    except:
                        pass
                    if (step_anim <= len(epoch_animations)):
                        step_anim += 1
                    else:
                        step_anim = 0

                    # debug information
                    console.log('%s Epoch: %d/%d | ⛔ Error: %f' % (current_anim, i+1, epochs, err), self.debug, True)
                    self.counter = 0
                else:
                    if (counter > epochs):
                        console.log('🟠 Counter value bigger than number of epochs... ', self.debug, True)
        
            except:
                self.cancelled = True
                console.log('🔴 Training cancelled! Check always your inputs!', self.debug)
                break
        
        if (self.cancelled == False):
            # training time ended
            self.training_time = (timer()-start)
            self.cpu_speed = epochs / self.training_time
            second_or_seconds = ' seconds'
            if (int(self.training_time) == 1): second_or_seconds = second_or_seconds[:-1]
            console.log('\n>> 🔵 Network trained in '+str(int(self.training_time))+second_or_seconds, self.debug)
            console.log('🔲 Average CPU speed: '+str(int(self.cpu_speed))+' (EPS)', self.debug)