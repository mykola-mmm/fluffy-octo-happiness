import tensorflow as tf

class WarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps, min_learning_rate):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule_fn = decay_schedule_fn
        self.warmup_steps = warmup_steps
        self.min_learning_rate = min_learning_rate

    def __call__(self, step):
        # Convert to float32 to ensure compatibility
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        
        # Linear warmup
        warmup_percent = step / warmup_steps
        warmup_lr = self.min_learning_rate + (self.initial_learning_rate - self.min_learning_rate) * warmup_percent
        
        # Decay schedule after warmup
        decay_lr = self.decay_schedule_fn(step - self.warmup_steps)
        
        # Return warmup_lr for steps < warmup_steps, else decay_lr
        return tf.cond(step < warmup_steps,
                      lambda: warmup_lr,
                      lambda: decay_lr)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "min_learning_rate": self.min_learning_rate,
        }
