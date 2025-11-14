# CoverHunterMPS

Fork of [Liu Feng's CoverHunter](https://github.com/Liu-Feng-deeplearning/CoverHunter) project. Goals: Make it run, and run fast, on any platform. Document it better. And build it out as a useful toolset for music research generally.

See https://ar5iv.labs.arxiv.org/html/2306.09025 for the July 2023 research paper that accompanied the original CoverHunter code. From their abstract: 

> Cover song identification (CSI) focuses on finding the same music with different versions in reference anchors given a query track. In this paper, we propose a novel system named CoverHunter that overcomes the shortcomings of existing detection schemes by exploring richer features with refined attention and alignments. [...] Experiments on several standard CSI datasets show that our method significantly improves over state-of-the-art methods [...].

The CoverHunterMPS project also has longer-term goals to expand the utility of the CoverHunter model to address a wide range of musicological questions and needs, such as:
- Identify known repertoire items in new, unfamiliar audio (basic CSI)
- Discover and describe how to adapt training hyperparameters for specific musical cultures.
- Make it easy for ethnomusicologists to train this model for specific musical cultures.
- Adapt this model to go beyond CSI to learn and classify audio using other musical categories such as rhythms, styles, tunings, forms, etc.
- Modify, confirm, or debunk established but currently merely subjectively defined musical concepts within specific musical cultures.

# Help Wanted

Collaborators are welcome at any time! That includes:
- Python co-authors
- Neural network designers
- Data scientists
- Data source contributors (music data)
- Musicologists (posing valuable challenges to tackle)
- Anyone interested in learning and practicing in the above fields 

Get started by participating in the Issues or Discussions tabs here in this Github site. Or contact Alan Ng directly (such as by using the Feedback Form link at the bottom of https://www.irishtune.info/public/MLdata.htm).

# Requirements

1. GPU-equipped computer. CPU-only hardware should work but will be very slow. Tested platforms:
    1. Apple computer with an Apple M-series chip
    2. Other computer with an Nvidia GPU (including free cloud options like Google Colab)
2. python3 (minimum version 3.10, tested on 3.11)

# Usage

Either:
- Clone this repo or download it to a folder on your computer. Run the following OS command-line commands from that folder. These commands assume you have a Unix/Linux/MacOS environment, but Windows equivalents exist - [see issue #10](https://github.com/alanngnet/CoverHunterMPS/issues/10).

- Or run this project in Google Colab, using this Colab notebook:
https://colab.research.google.com/drive/1HKVT3_0ioRPy7lrKzikGXXGysxZrHpOr?usp=sharing

## Install Requirements

1. The requirements.txt file contains the python dependencies of the
   project. Run `python -m pip install requirements.txt` to install, or `make
   virtualenv` to install the requirements in a virtualenv (the python3-venv package
   must be installed).

2. Install the `sox` package and its libraries. In some distributions, those
   libraries come in a separate package, like `libsox-fmt-all`.

## Data Preparation

Follow the example of the prepared Covers80 dataset included with the original CoverHunter. Directions here are for using that prepared data. See also the "dataset.txt format" heading below.

1. Download and extract the contents of the `covers80.tgz` file from http://labrosa.ee.columbia.edu/projects/coversongs/covers80/
2. Abandon the 2-level folder structure that came inside the covers80.tgz file, flattening so all the .mp3 files are in the same folder. One way to do this is:
    1. In Terminal, go to the extracted `coversongs` folder as the current directory. Then: 
    2. `cd covers32k && mv */* .; rm -r */`
3. Convert all the provided .mp3 files to .wav format. One way to do this is:
    1. `setopt EXTENDED_GLOB; for f in *.mp3; do sox "$f" -r 16000 "${f%%.mp3}.wav" && rm "$f"; done`
4. Move all these new .wav files to a new folder called `wav_16k` in the project's `data/covers80` folder.
5. You can delete the rest of the downloaded `covers80.tgz` contents.

Background explanation: Covers80 is a small, widely used dataset of modern, Western pop music intended only for benchmarking purposes, so that the accuracy of different approaches to solving the problem of CSI can be compared against each other. It is far too small to be useful for neural-network training, but it is a rare example of a published, stable collection of audio files. This makes it easy for you to get started, so you can confirm you have a working setup of this project without having to have your own set of audio files and their metadata ready. You might even end up using Covers80 yourself as a benchmarking test to see how well your own training project handles modern, Western pop music in comparison to published Covers80 benchmarks from other CSI projects.

## Feature Extraction

You must run this before proceeding to the Train step. And you can't run this without first doing the Data Preparation step above. See "Input and Output Files" below for more information about what happens here. In summary, this step generates some data augmentation - artificial variants of the real music you provide that help the neural network generalize across the various ways that humans might perform any musical work -, converts all of that audio (original and artificial) to CQT arrays (basically a type of spectrogram), and does some plain old data wrangling to prepare the metadata that the training script will need. 

To use the Covers80 example you prepared above, next run this from the project root folder:

`python3 -m tools.extract_csi_features data/covers80/`


## Training

Training is the core of the work that your *computer* will do to learn how to differentiate the various works and performances of music you give it. When successfully done, it will have constructed a general model which only needs to be trained once for your targeted musical culture. It will then be able to distinguish this culture's musical works from each other even when it never encountered those particular works during training. See the "Reference embeddings" section below regarding what life will look like once you reach that end goal of training.

Before you get to that ideal future state, the core of the work that *you* will do will be preparing your training data and discovering the optimal training hyperparameters. See the "Training Hyperparameters" section below. 

The training script's output consists of checkpoint files and embedding vectors, described below in the "Training Checkpoint Output" section.

Note: Don't use the `torchrun` launch command offered in original CoverHunter. At least in a single-computer Apple Silicon context, it is not only irrelevant, it actually slows down performance. In my tests on an MPS computer, `torchrun` slowed down tools.train performance by about 20%.

### Training example using Covers80

The original CoverHunter project included a prepared configuration to run a training session on the Covers80 dataset, and this is now located in the 'training/covers80' subfolder of this project. See the "Background explanation" above in the Data Preparation section about what to expect from using Covers80 for training. In particular, their test configuration used the same dataset for both training and validation, so results looked fabulously accurate and were essentially meaningless except that you could confirm that your setup is working. This fork added a train/validate/test data-splitting function in the extract_csi_features tool, along with corresponding new data-preparation hyperparameters, so you can choose to try more realistic training - in which the model validates its learning against data it has not seen before - even if you only have Covers80 data to play with.

You may need to edit the training hyperparameters in the `hparams.yaml` configuration file in the folder `training/covers80/config` before starting a training run. For example:
* To skip the step of also setting up covers80 not only as your training data but also in its normal use as your covers80 testset, just comment out the 4 `covers80` lines of your `hparams.yaml`. If you do want to see the surreal perfection of covers80 training vs covers80 testset, then follow the instructions in step 2 of the Evaluation section below to create your `data/covers80_testset/full.txt` file.
* Or configure the `testset` lines to point to any other testset(s) you like, for example the other testsets cited in the original CoverHunter paper, or the Irish traditional ones published at https://www.irishtune.info/public/MLdata.htm . See [Training Hyperparameters](https://github.com/alanngnet/CoverHunterMPS#data-sources) for details. 
* If you run into memory limits, start with decreasing the batch size from 64 to 32.

The one required command-line parameter for the training script is to specify the path where the training hyperparameters are available and where the model output will go, like this:

`python -m tools.train training/covers80/`

This fork also added an optional `--runid` parameter so you can distinguish your training runs in TensorBoard in case you are experimenting:

`python -m tools.train training/covers80/ --runid 'first try'`

To see the TensorBoard live visualization of the model's progress during training, run this in a separate terminal window, from the root of the project folder, and then use the URL listed in the output to watch the TensorBoard:

`tensorboard --logdir=training/covers80/logs`

## Hyperparameter Tuning

After you use the tools.train script to confirm your data is usable with CoverHunterMPS, and perhaps to do some basic experimentation, you may be motivated to experiment with a wide range of training hyperparameters to discover the optimal settings for your data that will lead you to better training metrics. You should be able to use your knowledge of its unique musical characteristics to make some educated guesses on how to diverge from the default CoverHunter hyperparameters, which were optimized for Western pop music.

Step 1: Study the explanations in the Training Hyperparameters section below to make some hypotheses about alternative hyperparameter values to try with your data. 

Step 2: Add your hypotheses as specific hyperparameter values to try in the hp_tuning.yaml file in the model's training folder, following the comments and examples there. 

Step 3: Launch training with `model_dir` as the one required parameter:

`python -m tools.train_tune training/covers80`

This script will not retain any model checkpoints from the training runs, but it does create separate log files for each run that you can monitor and study in TensorBoard.

If you are running on a CUDA platform, the `make_deterministic()` function in tools.train_tune may have significant performance disadvantages for you. Consider whether you'd rather comment out that line and instead run enough different random seeds to compensate for non-deterministic training behavior so that you can reliably compare results between different hyperparameter settings.

Tip for deep learning newbies: A good AI assistant can help greatly with hyperparameter tuning advice. Give it this project's files, the hyperparameters you tried, and the corresponding screenshots of your resulting Tensorboard validation loss and testset mAP metrics. Then ask it for advice on what to try next. 

## Evaluation

This script evaluates your trained model by providing standard mAP (mean average precision) and MR1 (mean rank one) training metrics, plus an optional t-SNE clustering plot (compare Fig. 3 in the CoverHunter paper).

1. Have a pre-trained CoverHunter model's output checkpoint files available. You only need your best set (typically your highest-numbered one). If you use original CoverHunter's pre-trained model from https://drive.google.com/file/d/1rDZ9CDInpxQUvXRLv87mr-hfDfnV7Y-j/view), unzip it, and move it to a folder that you specify in step 3 below.
2. Run your query data through `extract_csi_features.py`. In the `hparams.yaml` file for the feature extraction, turn off all augmentation. See `data/covers80_testset/hparams.yaml` for an example configuration to treat covers80 as the query data:<br> `python3 -m tools.extract_csi_features data/covers80_testset`<br>
The important output from that is `full.txt` and the `cqt_feat` subfolder's contents.
3. Run the evaluation script. This example assumes you are using the trained model you created in `training/covers80` and you want to use all the optional features I added in this fork:<br>
`python3 -m tools.eval_testset training/covers80 data/covers80_testset/full.txt data/covers80_testset/full.txt -plot_name="training/covers80/tSNE.png" -dist_name='distmatrix' -test_only_labels='data/covers80/test-only-work-ids.txt'`

See the "Training checkpoint output" section below for a description of the embeddings saved by the `eval_for_map_with_feat()` function called in this script. They are saved in a new subfolder of the `pretrained_model` folder named `embed_NN_tmp` where NN is the highest-numbered epoch subfolder in the `pretrained_model` folder.

### Arguments

#### query_in_ref_path
CoverHunter only shared an evaluation example for the case when query and reference data are identical, presumably to do a self-similarity evaluation of the model. But there is an optional 4th parameter for `query_in_ref_path` that would be relevant if query and reference are not identical. See the "query_in_ref" heading below under "Input and Output Files."

#### plot_name
The optional `plot_name` argument is a path or just a filename where you want to save the t-SNE plot output. If you provide just a filename, `model_dir` will be used as the path. See example plot below. Note that your query and reference files must be identical to generate a t-SNE plot (to do a self-similarity evaluation).

#### test_only_labels
The optional `test_only_labels` argument is a path to the text file generated by `extract_csi_features.py` if its hyperparameters asked for some work_ids to be reserved exclusively for the test dataset. The t-SNE plot will then mark those for you to see how well your model can cluster classes (work_ids) it has never seen before.

This figure shows the results of training from scratch on the covers80 dataset with a train/val/test split of 8:1:1 and 3 classes (work_ids) reserved exclusively for the test dataset.
![t-SNE plot for Covers80](tSNE-example.png)

#### dist_name
The optional `dist_name` argument is a path where you want to save the distance matrix and ref labels so that you can study the results separately, such as perhaps doing custom t-SNE plots, etc.

#### marks
The default value for the optional `marks` argument is 'markers', which makes the output for `plot_name` differentiate works by using using standard matplotlib markers in various colors and shapes. The alternative value is 'ids' which uses the `work_id` numbers defined by extract_csi_features instead of matplotlib markers.   

## Production Training

Once you have tuned your data and your hyperparameters for optimal training results, you may be ready to train a model that knows *all* of your data, without reserving any data for validation and test sets. The tools/train_prod.py script uses stratified K-fold cross validation to dynamically generate validation sets from your dataset so that the model is exposed to all works and perfs equally. It concludes with one final training run on the entire dataset in which the dataset you specify in `test_path` serves as the validation dataset (for early stopping purposes). This final validation set should be entirely unseen perfs, even if some or all of the works are represented in the training data.

Use the `full.txt` output from `extract_csi_features.py` for your `train_path` with `val_data_split`, `val_unseen`, `test_data_split`, and `test_data_unseen` all set to 0. Prepare the `training/covers80/hparams_prod.yaml` file following the instructions in the comment header of `train_prod.py`. An example `hparams_prod.yaml` is provided for using covers80 for testing purposes.

You may need to experiment with learning rates and other hyperparameters for the somewhat different training situation of training on your full dataset if your hyperparameter tuning work used significantly smaller datasets. Also consider experimenting with the hard-coded learning-rate strategy for later folds after the first fold that is configured within `train_prod.py` in the `cross_validate()` function. Look for the comment line "# different learning-rate strategy for all folds after the first."

Launch training with:

`python -m tools.train_prod training/covers80/ --runid='test of production training'`

TensorBoard will show each fold as a separate run, but within a continuous progression of epochs. You can safely interrupt production training for any reason and re-launching it with the same command will resume from the last fold and checkpoint that was automatically saved by this script.

## Generate Reference Embeddings

After you have trained a model and are satisfied with its quality based on the metrics you saw during training and from the evaluation script, it's time to use your model to generate reference embeddings. An embedding is a numerical representation generated by your trained model of any audio sample, essentially identifying the audio in a high-dimensional conceptual space that differentiates works from each other based on the knowledge the neural network learned from your training data. Your trained model can even generate embeddings for recordings that were not used in training, assuming the new recordings fit well inside the same musical culture and vocabulary as the one you used in training. 

Reference embeddings, then, are the complete set of embeddings for all of the recorded performances you would like to be already known to your final inference solution. These points in space, like stars in a galaxy, can then be compared with a new embedding from new audio, and by measuring the distance between the new embedding to all the reference embeddings, you can locate the new audio in that galaxy, by learning who the nearest neighbors are.

Example for covers80:

`python -m tools.make_embeds data/covers80 training/covers80`

See comments at the top of the make_embeds script for more details. The output of `make_embeds` is `reference_embeddings.pkl`.

### Centroids Option

The embeddings generated above represent all of the individual _performances_ you provided to the `make_embeds` script. What if you are instead interested in simplifying that galaxy of performances as the galaxy of _works_ instead, where one embedding represents the average location of all the performances assigned to that work? Then use the `compute_centroids.py` utility as a next step. Example for covers80:

`python -m tools.compute_centroids data/covers80/reference_embeddings.pkl data/covers80/work_centroids.pkl`

## Inference (work identification)

Now that you have reference embeddings and the trained model to generate new embeddings for any new audio, you can use the `identify` script to identify any music you give it. See the high-level explanation of how this works in the "Generate reference embeddings" section above. See comments at the top of tools.identify for documentation of the parameters.
 
Example for covers80:

`python -m tools.identify data/covers80 training/covers80 query.wav -top=10`

To interpret the output, use the data/covers80/work_id.map text file to see which `work_id` goes with which `work`. Good news: even the bare-bones demo of training from scratch on covers80 shows that CoverHunter does a good job of identifying versions (covers) of those 80 pop songs.

Optional parameter to save the embedding as a NumPy array:

`python -m tools.identify data/covers80 training/covers80 query.wav -save query.npy`

Future goal and call for help: How do we take this command-line solution for inference and productionize it for broader use outside the context of the specific machine where this CoverHunterMPS project was installed?

### Centroids Option

If you chose to generate work embeddings using `compute_centroids`, you can optionally have the `identify` script return matching works instead of matching performances. Use the `-centroids` flag like this:

`python -m tools.identify data/covers80 training/covers80 query.wav -top=10 -centroids`

See the comments just before the `if radii and args.centroids:` line of the script to also consider whether or not to use the confidence-radius filtering option, and then tune the `min_threshold` and `padding` constants for your model based on running query experiments to get your desired inference results.

## Adding New Works to a Model's Knowledge

Once you have a well-trained model that performs well with real-world inference to match to performances stored in your `references_embeddings.pkl` file, you can expand your model's vocabulary of works it can identify without having to re-train the entire model.

1. Build a "diff" aka "delta" metadata file suitable for input to `extract_csi_features` describing the new performances to process, including their human-identified work_ids, and where the corresponding audio files are located. 
2. Feed that to `extract_csi_features` which generates .npy CQT files and the full.text metadata file, and configure its hyperparameters to do no augmentation.
3. Merge the output (the diff `full.txt` and its CQT files) with your master full.txt and its CQT files.
4. Run `make_embeds.py` on the newly merged `full.txt` to update your `reference_embeddings.pkl` file that `identify.py` needs. 

Then you can resume using `identify.py` and it will "know" the new works and performances you added.

## Coarse-to-Fine Alignment Training

CoverHunter did not include an implementation of the coarse-to-fine alignment training described in the research paper. (Liu Feng confirmed to me that his employer considers it proprietary technology). [See issue #1](https://github.com/alanngnet/CoverHunterMPS/issues/1). But it did include this script which apparently could be useful as part of an implementation we could build ourselves. The command to launch the alignment script that CoverHunter included is:

`python3 -m tools.alignment_for_frame pretrained_model data/covers80/full.txt data/covers80/alignment.txt`

Arguments to pass to the script:
1. Folder containing a pretrained model. For example if you use original CoverHunter's model from https://drive.google.com/file/d/1rDZ9CDInpxQUvXRLv87mr-hfDfnV7Y-j/view), unzip it, and move it to a folder that you rename to `pretrained_model` at the top level of your project folder. That folder in turn must contain a `checkpoints` subfolder that contains the do_000[epoch] and g_000[epoch] checkpoint files.
2. The output from tools/extract_csi_features.py or an equivalent script. The metadata file like full.txt  must include `work_id` values for each `perf` (unlike the raw `dataset.txt` file that CoverHunter provided for covers80).
3. The `alignment.txt` file will receive the output of this script.

# Input and Output Files

## Hyperparameters (hparams.yaml)

There are two different hparams.yaml files, each used at different stages. 

### Data Preparation Hyperparameters

The hparams.yaml file located in the folder you provide on the command line to tools.extract_csi_features.py is used only by that script.

| key | value |
| --- | --- |
| add_noise| Original CoverHunter provided the example of: <div>{<br> &nbsp; `prob`: 0.75,<br> &nbsp; `sr`: 16000,<br> &nbsp; `chunk`: 3,<br> &nbsp; `name`: "cqt_with_asr_noise",<br> &nbsp; `noise_path`: "dataset/asr_as_noise/dataset.txt"<br>}<br>However, the CoverHunter repo did not include whatever might supposed to be in "dataset/asr_as_noise/dataset.txt" file nor does the CoverHunter research paper describe it. If that path does not exist in your project folder structure, then tools.extract_csi_features will just skip the stage of adding noise augmentation. At least for training successfully on Covers80, noise augmentation doesn't seem to be needed.|
| aug_speed_mode | list of ratios used in tools.extract_csi_features for speed augmention of your raw training data. Example: [0.8, 0.9, 1.0, 1.1, 1.2] means use 80%, 90%, 100%, 110%, and 120% speed variants of your original audio data.|
| bins_per_octave | See `fmin` and `n_bins`. If your musical culture uses a scale that does not fit in the Western standard 12-semitone scale, set this to a higher number. Default 12. |
| device | 'mps' or 'cuda', corresponding to your GPU hardware and PyTorch library support. 'cpu' is not currently implemented but could be if needed. Original CoverHunter used CPU for this stage but was much slower. |
| fmin | The lowest frequency you want the CQT arrays to include. Set this to just below the lowest pitch used in the musical culture you are teaching the model. Consider only the pitches relevant to the work-identification skill you want it to learn. For example, in some cultures, bass accompaniment is not relevant for work identification. Default is 32. |
| n_bins | The number of frequency bins you want the CQT arrays to include. For example, if you set `bins_per_octave` to 12, then set `n_bins` to 12 times the number of octaves above `fmin` that are relevant to this culture's work-identification skill. Be sure to also set the `input_dim` training hyperparameter to match this number. Default is 96. |
| val_data_split | percent of training data to reserve for validation expressed as a fraction of 1. Example for 10%: 0.1 |
| val_unseen | percent of work_ids from training data to reserve exclusively for validation expressed as a fraction of 1. Example for 2%: 0.02 |
| test_data_split | percent of training data to reserve for test expressed as a fraction of 1. Example for 10%: 0.1 |
| test_data_unseen | percent of work_ids from training data to reserve exclusively for test expressed as a fraction of 1. Example for 2%: 0.02  |

### Training Hyperparameters
The hparams.yaml file located in the "config" subfolder of the path you provide on the command line to tools.train.py uses all the other parameters listed below during training.

#### Data Sources

| key | value |
| --- | --- |
| covers80:<br> &nbsp; query_path<br> &nbsp; ref_path<br> &nbsp; every_n_epoch_to_test | Test dataset(s) used for automated model evaluation purposes during training. "covers80" was the only example provided with the original CoverHunter. For an example of a different culture's test set, see https://www.irishtune.info/public/MLdata.htm. Note that ref_path and query_path are set to the same data in order to do a self-similarity evaluation, testing how well the model can cluster samples (perfs) relative to their known classes (works). You can add as many test datasets as you want. Each will be displayed as separate results in the TensorBoard visualization during training.<br><br>New testsets must be added to the `src/trainer.py` script in the list where `ALL_TEST_SETS` is defined.<br><br>Subparameters for covers80:<br>`query_path`: "data/covers80/full.txt"<br>`ref_path`: "data/covers80/full.txt"<br>`every_n_epoch_to_test`: How many epochs to wait between each test of the current model against this testset. |
| test_path | Compare `train_path` and `val_path`. This dataset is used in each epoch to run the same validation calculation as with the `val_path`. Presumably one should include both classes and samples that were excluded from both `train_path` and `val_path`. |
| train_path | path to a JSON file containing metadata about the data to be used for model training (See full.txt below for details) |
| val_path | Path to a JSON file containing metadata about the data to be used for model validation. Compare `test_path` above. Presumably one should include a balanced distribution of samples that are *not* included in the `train_path` dataset, but do include samples for the classes represented in the `train_path` dataset. (See full.txt below for details) |

#### Dataset Parameters
| key | value |
| --- | --- |
| chunk_frame | List of 3 numbers used with `mean_size` that describe the duration of each chunk, measured as a count of CQT features. CoverHunter's covers80 config used [1125, 900, 675]. Here the word "chunk" apparently refers to the chunks described in the time-domain pooling strategy part of the CoverHunter paper, not the chunks discussed in their coarse-to-fine alignment strategy. See also `chunk_s`. In our experiments, the 5:4:3 ratio that CoverHunter used is significantly better than a variety of alternative ratios we tried. However, in Irish traditional music, which has shorter time structures than Western pop music, we achieved better results using shorter durations than [1125, 900, 675]. | 
| chunk_s | Duration of the first-listed (longest) `chunk_frame` in seconds. You have to manually calculate `chunk_s` = `chunk_frame[0]` / audio sample rate * `mean_size`. Couldn't the script just calculate this itself using CQT hop-size to get the sample rate? |
| cqt: hop_size: | Fine-grained time resolution, measured as duration in seconds of each CQT spectrogram slice of the audio data (the inverse of the audio sample rate). CoverHunter's provided setting is 0.04 with a comment "1s has 25 frames", but this meaning of "frame" is not the same meaning of "frame" as used more appropriately in `chunk_frame`. Presumably the intended meaning here would conventionally be described as: "audio sample rate of 25 samples per second." The value 25 is hard-coded as an assumption into CoverHunter in various places. |
| data_type | "cqt" (default) or "raw" or "mel". It remains unknown whether the CoverHunter team actually implemented or tested anything but CQT-based training. |
| mean_size | See `chunk_s` above. An integer used in combination with `chunk_frame` to define the length of the chunks. |
| mode | "random" (default) or "defined". Changes behavior of AudioFeatDataset related to how it cuts each audio sample into chunks. "random" is described in CoverHunter code as "cut chunk from feat from random start". "defined" is described as "cut feat with 'start/chunk_len' info from line." We observed better training results using "defined" when working with datasets that are very consistently trimmed so that CSI-relevant audio always starts right at the beginning of the recording. "random" would be better when CSI-irrelevant audio may be present at the start of many of your audio data samples. |
| m_per_class | From CoverHunter code comments: "m_per_class must divide batch_size without any remainder" and: "At every iteration, this will return m samples per class. For example, if dataloader's batch-size is 100, and m = 5, then 20 classes with 5 samples iter will be returned." |
| spec_augmentation | Spectral augmentation settings, used to generate temporary data augmentation on the fly during training.  CoverHunter settings were:<br>`random_erase`:<br> &nbsp; `prob`: 0.5<br> &nbsp; `erase_num`: 4<br>`roll_pitch`:<br> &nbsp; `prob`: 0.5<br> &nbsp; `shift_num`: 12 |
| spec_augmentation : random_erase | During each epoch, each CQT array may have a rectangular block of its array values replaced with the value -80 (a low amplitude signal). The size of the block is defined as 25% of the height of the frequency bins and 10% of the width of the time bins. `prob` specifies the probability of calling the erase method for this feature in this epoch, between 0 and 1. `erase_num` specifies the quantity of such blocks that will be erased if the erase method is called. `region_size` specifies the size of each erased block, as (width, height) as fractions of the CQT array size. Default is "[.25, .1]"|
| spec_augmentation : roll_pitch | During each epoch, each CQT array may be shifted pitch-wise. CoverHunter's original method, left as the default here, was to rotate the entire array in the frequency dimension, with the overflowing content wrapped around to the opposite end of the spectrum. For example, if shifted an octave up, then the top octave's CQT content would be presented as the bottom octave of content. `prob` specifies the probability of doing this for this feature in this epoch, between 0 and 1. `shift_num` specifies the number of frequency CQT bins by which the array will be shifted. `method` accepts 3 values: 1) "default" is the original CoverHunter method 2) "low_melody" is an alternative approach added for CoverHunterMPS to accommodate musical cultures in which CSI-significant melodic content may appear in the bottom frequency range of the CQT array. Since trimming CQT arrays to eliminate irrelevant harmonic and percussive content in the bottom octaves has proven beneficial, this feature can be significantly useful. In this case, instead of rotating the entire array either up or down, the array is shifted upwards either 1 x or 2 x `shift_num` bins, and overflowing high-frequency content is simply discarded, instead of being copied to the bottom rows of the array. 3) "flex_melody" generalizes the "low_melody" approach by using loudness to estimate where the tonal center is, in order to avoid shifting the melody off the "edge" of the spectrogram either too low or too high. |

#### Training Parameters
| key | value |
| --- | --- |
| adam_b1 and adam_b2 | See https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html for documentation of the two "beta" parameters used by the AdamW optimizer that the CoverHunter authors chose. Our experiments showed these can have a strong impact. Note that the CoverHunter default values of .8 and .99 are not the usual default AdamW values, for unknown reasons. We recommend experimenting with these values. |
| batch_size | Usual "batch size" meaning in the field of machine learning. An important parameter to experiment with. Original CoverHunter's preset batch size of 16 was no longer able to succeed at the covers80 training task after @alanngnet fixed an important logic error in extract_csi_features.py. Now only batch size 32 or larger works for covers80. Be sure to consider adjusting `learning_rate` and `lr_decay` whenever you change `batch_size` based on general deep-learning best practices and your own experimentation. Batch size is traditionally set to an exponent of 2, but in practice could be any integer multiple of your `m_per_class` hyperparameter. |
| device | 'mps' or 'cuda', corresponding to your GPU hardware and PyTorch library support. Theoretically 'cpu' could work but untested and probably of no value. |
| early_stopping_patience | how many epochs to wait for validation loss to improve before early stopping |
| learning_rate | The initial value for how much variability to allow the model during each learning step. See `lr_decay`. Default = .001. |
| lr_decay | Learning-rate decay - see `learning_rate`. Default = .9975, but for small data sets, such as during testing and tuning work, we found lower values like .99 more appropriate. |
| min_lr | Minimum learning rate, below which `lr_decay` is ignored. Default = 0.0001. |

#### Model Parameters
Traditionally these model dimensions are restricted to exponents of 2 (32, 64, 128, etc.) but in practice other values may work well. Experimentation is necessary if you diverge from those multiples to avoid performance disadvantages from potential memory-allocation inefficiencies in your particular environment, while seeking higher quality results from higher dimensions.

| key | value |
| --- | --- |
| embed_dim | Generally matches `encoder : output_dims` but you can set this to a higher value than `output_dims` if your output_dims are at the limit of the "curse of dimensionality" in order to gain more complex learning without sacrificing the value of inference embeddings for cosine-similarity-based clustering. Default 128. |
| encoder | # model-encode<br>Subparameters:<br>`attention_dim`: 256 # "the hidden units number of position-wise feed-forward"<br>`output_dims`: This defines the dimensionality of the final embeddings the model generates, which impacts your results using the `identify.py` tool. Default 128, which is low enough to avoid the "curse of dimensionality."<br>`num_blocks`: 6 # number of decoder blocks |
| input_dim | The "vertical" (frequency) dimension size of the CQT arrays you provide to the model. Set this to the same value you used for `n_bins` in the data preparation hyperparameters. Default is 96. |


## dataset.txt

A JSON formatted or tab-delimited key:value text file (see format defined in the utils.py::line_to_dict() function) expected by extract_csi_features.py that describes the training audio data, with one line per audio file.
| key | value |
| --- | --- |
| perf | Unique identifier. Abbreviation for "performance." CoverHunter originally used "utt" throughout, borrowing the term "utterance" from speech-recognition ML work which is where much of their code was adapted from. Example "cover80_00000000_0_0". |
| wav | relative path to the raw audio file. Example: "data/covers80/wav_16k/annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.wav" |
| dur_s | duration of the audio file in seconds. Example 316.728 |
| work | title of the work. Example "A_Whiter_Shade_Of_Pale" The `_add_work_id()` function in extract_csi_features assumes that this string is a unique identifier for the work (so it can't handle musically distinct works that happen to have the same title). Advice: Use a unique, stable identifier that applies across the entire musical culture in which you will be training. For example in Irish traditional music, use the irishtune.info TuneID number. |
| version | Not used by CoverHunter. Example from covers80: "annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.mp3", which would have been the original audio file source for that perf. |

## full.txt 

full.txt is the JSON-formatted training data catalog for tools.train.py, generated by tools.extract_csi_features. In case you do your own data prep instead of using tools.extract_csi_features, here's the structure of full.txt.

| key | value |
| --- | --- |
| perf | See dataset.txt. Except in this context, for each original perf, extract_csi_features generates additional artificial variants, which each get their own perf identifier. |
| wav | (see dataset.txt) |
| dur_s |(see dataset.txt) |
| work | (see dataset.txt) |
| version | (see dataset.txt) |
| feat | path to the CQT features of this perf stored as .npy array. Example: "data/covers80/cqt_feat/sp_0.8-cover80_00000146_71_0.cqt.npy" |
| feat_len | output of len(np.load(feat)). Example: 9198 |
| work_id | internal, arbitrary unique identifier for the work. This is what teaches the model which perfs (performances) are considered by humans to be the "same work." Example: 0 |
| version_id | internal, arbitrary unique identifier for each artificially augmented variant of the original perf (performance). Example: 0 |

## work_id.map 

Text file crosswalk between "work" (unique identifying string per work) and the "work_id" number arbitrarily assigned to each "work" by the extract_csi_features.py script. Not used by any scripts in this project currently, but definitely useful as a reference for human interpretation of training results.


## Other Files Generated by extract_csi_features.py

| filename | comments |
|---|---|
| cqt_feat subfolder | Contains the Numpy array files of the CQT data for each file listed in full.txt. Needed by train.py. Also used each time you run extract_csi_features.py to save time in creating CQT data by skipping CQT generation for samples already represented in this folder. |
| data.init.txt | Copy of dataset.txt after sorting by `perf` and de-duping. Not used by train.py |
| test.txt | A subset of full.txt generated by the `_split_data_by_work_id()` function intended for use by train.py as the `test` dataset. |
| test-only-work-ids.txt | Text file listing one work_id per line for each work_id that the train/val/test splitting function held out from train/val to be used exclusively in the test dataset. This file can be used by `eval_testset.py` to mark those samples in the t-SNE plot. |
| full.txt | See above detailed description. Contains the entire dataset you provided in the input file. | 
| work_id.map | Text file, with 2 columns per line, separated by a space, sorted alphabetically by the first column. First column is a distinct "work" string taken from dataset.txt. Second column is the `work_id` value assigned to that "work." |
| sp_aug subfolder | Contains the sox-modified wav speed variants of the raw training .wav files, at the speeds defined in hparams.yaml. Not used by train.py. Also used each time you run extract_csi_features.py to save time in creating speed variants by skipping speed augmentation for samples already represented in this folder. |
| sp_aug.txt | Copy of data.init.txt but with addition of 1 new row for each augmented variant created in sp_aug/*.wav. Not used by train.py. |
| train.txt | A subset of full.txt generated by the `_split_data_by_work_id()` function intended for use by train.py as the `train` dataset. |
| val.txt | A subset of full.txt generated by the `_split_data_by_work_id()` function intended for use by train.py as the `val` dataset. |


Original CoverHunter also generated the following files, but were not used by their published codebase, so I commented out those functions:

| filename | comments |
|---|---|
| work_id_num.map | Text file, not used by train.py, maybe not by anything else? |
| work_name_num.map | Text file, not used by train.py, maybe not by anything else? |

## CQT Array Structure

The structure of the CQT arrays as handled within this project is:
[time bins ordered from start to end, frequency bins ordered from low to high frequencies]

Note that to visualize these arrays in traditional spectrogram form with time on the x axis and frequency on the y axis, the CQT arrays must be transposed, for example by using the native Python `.T` suffix.

## Training Checkpoint Output

Using the default configuration, training saves checkpoints after each epoch in the training/covers80 folder.

The `checkpoints` subfolder gets two files per epoch: do_000000NN and g_000000NN where NN=epoch number. The do_ files contain the AdamW optimizer state. The g_ files contain the model's state dictionary. "g" might be an abbreviation for "generator" given that a transformer architecture is involved?

The `eval_for_map_with_feat()` function, called at the end of each epoch, also saves data in a separate new subfolder for each epoch, named epoch_NN_covers80. This in turn gets a `query_embed` subfolder containing the model-generated embeddings for every sample in the training data, plus the embeddings for time-chunked sections of those samples, named with a suffix of ...__start-N.npy where N is the timecode in seconds of where the chunk starts. The saved embeddings are 1-dimensional arrays containing 128 double-precision (float64) values between -1 and 1. The epoch_NN_covers80 folder also gets an accompanying file `query.txt` (with an identical copy as `ref.txt`) which is a text file listing the attributes of every training sample represented in the `query_embed` subfolder, following the same format as described above for `full.txt`.

## query_in_ref

The file you can prepare for the `tools/eval_testset.py` script to pass as the 4th parameter `query_in_ref_path` (CoverHunter did not provide an example file or documentation) assumes:
- JSON or tab-delimited key:value format
- The only line contains a single key "query_in_ref" with a value that is itself a list of tuples, where each tuple represents a mapping between an index in the query input file and an index in the reference input file.
This mapping is only used by the `_generate_dist_matrix()` function. That function explains: "List[(idx, idy), ...], means query[idx] is in ref[idy] so we skip that when computing mAP." idx and idy are the sequentially assigned index numbers to each perf in the order they appear in the query and ref data sources.

# Code Map

Hand-made visualization of how core functions of this project interact with each other. Also includes additional beginner-friendly or verbose code-commenting that I didn't add to the project code. Not regularly maintained, but still useful for getting oriented in this project's code:

https://miro.com/app/board/uXjVNkDkn70=/ 

# Unit Tests

Unit tests are in progress, currently only with partial code coverage. Run them from
the repository root using:

`python3 -m unittest -c tests/test_*.py`

or if you installed the project in a virtualenv:

`make tests` 

# Distribution of Works vs. Performances 

As a contribution to the CSI community, where the [SHS100K dataset](https://github.com/NovaFrost/SHS100K) has been used as a standard training dataset for many years, including for the CoverHunter research paper, here is a histogram showing the distribution of works vs. performances in SHS100K.

![SHS100K Histogram](SHS100k_histogram.png)

This figure may be helpful as a reference for comparing the distribution of works vs. performances in datasets you want to use with CoverHunterMPS, knowing that CoverHunter was able to train successfully given this distribution. 

To help you understand this visualization of the SHS100K dataset, here are some example data points from it: The most common work ("Summertime") is represented by  387 performances, and there are over 300 works having only a single performance. The most common count of performances per work is 6.

To get a sense of the range of usable distributions, here's a different dataset's histogram, generated using the `tools/plot_histogram.py` utility in this project. Total count of performances was 26,482 across 6,606 works. The maximum count of performances for a single work was 72. This dataset was sufficient to achieve .97 mAP on the [reels50easy test](https://www.irishtune.info/public/MLdata.htm) by epoch 33. 

![irishtune.info v3.2 Histogram](irishtune.infov3.2_histogram.png)
