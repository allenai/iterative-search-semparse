# pylint: disable=too-many-lines
import logging
import time
from typing import Dict, Optional, List, Tuple, Union, Iterable

from overrides import overrides

import torch
import torch.optim.lr_scheduler



from allennlp.common.util import peak_memory_mb, gpu_memory_mb
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.trainer import Trainer, time_to_str

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# pylint: disable=no-member
class TrainerWithNoop(Trainer):
    """
    Adds an additional functionality of skipping "noop" batches
    - batches for which no training data was found. While this may sound counterintuitive,
    it is a possibility in, for example, Dynamic MML where the model wasn't able to find any logical
    form (spurious or not) that evaluated to the correct answer.
    """

    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False) -> None:
        """
        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``int``, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``PytorchLRScheduler``, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, this will use the ``validation_metric``
            provided to determine if learning has plateaued.  To support updating the learning
            rate on every batch, this can optionally implement ``step_batch(batch_num_total)`` which
            updates the learning rate given the batch number.
        summary_interval: ``int``, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : ``int``, optional, (default = ``None``)
            If not None, then log histograms to tensorboard every ``histogram_interval`` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            ``model.get_parameters_for_histogram_tensorboard_logging``.
            The layer activations are logged for any modules in the ``Model`` that have
            the attribute ``should_log_activations`` set to ``True``.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : ``bool``, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : ``bool``, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        """
        print("Using NOOP trainer")
        super().__init__(
                model,
                optimizer,
                iterator,
                train_dataset,
                validation_dataset,
                patience,
                validation_metric,
                validation_iterator,
                shuffle,
                num_epochs,
                serialization_dir,
                num_serialized_models_to_keep,
                keep_serialized_model_every_num_seconds,
                model_save_interval,
                cuda_device,
                grad_norm,
                grad_clipping,
                learning_rate_scheduler,
                summary_interval,
                histogram_interval,
                should_log_parameter_statistics,
                should_log_learning_rate)



    def _batch_loss(self, batch: torch.Tensor, for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batch and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = self._data_parallel(batch)
        else:
            batch = util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self._model(**batch)

        noop = False
        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self._model.get_regularization_penalty()
            if "noop" in output_dict:
                noop = output_dict["noop"]

        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None
        return loss, noop


    @overrides
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        logger.info(f"Peak CPU memory usage MB: {peak_memory_mb()}")
        for gpu, memory in gpu_memory_mb().items():
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self._model.train()

        # Get tqdm for the training batches
        train_generator = self._iterator(self._train_data,
                                         num_epochs=1,
                                         shuffle=self._shuffle)
        num_training_batches = self._iterator.get_num_batches(self._train_data)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        if self._histogram_interval is not None:
            histogram_parameters = set(self._model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        for batch in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self._log_histograms_this_batch = self._histogram_interval is not None and (
                    batch_num_total % self._histogram_interval == 0)

            self._optimizer.zero_grad()
            loss, noop = self._batch_loss(batch, for_training=True)
            if noop:
                continue


            loss.backward()

            train_loss += loss.item()

            batch_grad_norm = self._rescale_gradients()

            # This does nothing if batch_num_total is None or you are using an
            # LRScheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)

            if self._log_histograms_this_batch:
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self._model.named_parameters()}
                self._optimizer.step()
                for name, param in self._model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                       update_norm / (param_norm + 1e-7),
                                                       batch_num_total)
            else:
                self._optimizer.step()

            # Update the description with the latest metrics
            metrics = self._get_metrics(train_loss, batches_this_epoch)
            description = self._description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if batch_num_total % self._summary_interval == 0:
                if self._should_log_parameter_statistics:
                    self._parameter_and_gradient_statistics_to_tensorboard(batch_num_total, batch_grad_norm)
                if self._should_log_learning_rate:
                    self._learning_rates_to_tensorboard(batch_num_total)
                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"], batch_num_total)
                self._metrics_to_tensorboard(batch_num_total,
                                             {"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._log_histograms_this_batch:
                self._histograms_to_tensorboard(batch_num_total, histogram_parameters)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, time_to_str(int(last_save_time))), [], is_best=False
                )

        return self._get_metrics(train_loss, batches_this_epoch, reset=True)

    @overrides
    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self._model.eval()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self._iterator

        val_generator = val_iterator(self._validation_data,
                                     num_epochs=1,
                                     shuffle=False)
        num_validation_batches = val_iterator.get_num_batches(self._validation_data)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch in val_generator_tqdm:

            loss, _ = self._batch_loss(batch, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = self._get_metrics(val_loss, batches_this_epoch)
            description = self._description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        return val_loss, batches_this_epoch



Trainer.register("noop_trainer")(TrainerWithNoop)
