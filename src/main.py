# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import sys
import os # <-- Import os for path manipulation
import argparse
import codecs
import cv2
import Levenshtein
import numpy as np

# Assuming DataLoader, Model, Batch, preprocess are in the same src directory
# If they are in subdirectories, adjust the import path
try:
    from DataLoader import DataLoader, Batch
    from Model import Model, DecoderType
    from SamplePreprocessor import preprocess
except ImportError as e:
    print(f"Error importing local modules (DataLoader, Model, SamplePreprocessor): {e}")
    print("Ensure these files exist in the 'src' directory or adjust import paths.")
    sys.exit(1)


# --- Path Configuration (Corrected for running from src/) ---
# Get the directory containing main.py (which is .../src/)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from src/)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

class FilePaths:
    "filenames and paths relative to project root"
    fnCharList = os.path.join(PROJECT_ROOT, 'model', 'charList.txt')
    fnAccuracy = os.path.join(PROJECT_ROOT, 'model', 'accuracy.txt')
    # Model files expected within the 'model' directory by the Model class internally
    # We assume the Model class handles finding 'snapshot-X' etc. within PROJECT_ROOT/model/
    fnModel = os.path.join(PROJECT_ROOT, 'model') # Base directory for the model

    # Paths for data (if needed outside main execution block)
    fnTrain = os.path.join(PROJECT_ROOT, 'data/')
    fnInfer = os.path.join(PROJECT_ROOT, 'data', 'test.png') # Example inference file if needed
    fnCorpus = os.path.join(PROJECT_ROOT, 'data', 'hindi_vocab.txt')

print(f"[main.py] Project Root: {PROJECT_ROOT}")
print(f"[main.py] Character List Path: {FilePaths.fnCharList}")
print(f"[main.py] Model Base Path: {FilePaths.fnModel}") # Keep this print for info
# --- End Path Configuration ---


# --- Global Variables for Model and Character List ---
model_instance = None
character_list = None
# --- End Global Variables ---


# --- Function to Load Model Globally ---
def load_model_globally():
    """Loads the NN model and character list once."""
    global model_instance, character_list
    print("[main.py] Attempting to load model and character list globally...")

    # 1. Load Character List
    try:
        if not os.path.exists(FilePaths.fnCharList):
            print(f"[main.py] ERROR: Character list not found at {FilePaths.fnCharList}")
            return False
        with codecs.open(FilePaths.fnCharList, encoding="utf8") as f:
            character_list = f.read()
        print(f"[main.py] Character list loaded ({len(character_list)} chars).")
    except Exception as e:
        print(f"[main.py] ERROR: Failed to load character list: {e}")
        character_list = None # Ensure it's None on failure
        return False

    # 2. Load Model
    try:
        print("[main.py] Initializing Model class...")
        # Set mustRestore=True is typical for inference
        # --- !!! REMOVED model_path_base argument here !!! ---
        model_instance = Model(character_list, DecoderType.BestPath, mustRestore=True)
        print("[main.py] Model instance created and weights likely loaded (check Model class logs).")
        return True
    except Exception as e:
        print(f"[main.py] ERROR: Failed to initialize or restore model: {e}")
        import traceback
        print(traceback.format_exc())
        model_instance = None # Ensure it's None on failure
        return False

# --- End Model Loading Function ---


# --- Call Model Loading Function at Module Level ---
# This code runs ONCE when main.py is imported by upload.py
if load_model_globally():
    print("[main.py] Global model loading successful.")
else:
    print("[main.py] WARNING: Global model loading failed. Inference will not work.")
# --- End Global Loading Call ---


# --- Training and Validation Functions (Keep as is, not used by web) ---
def train(model, loader):
    "train NN"
    # (Your existing train code - unchanged)
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best valdiation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = 5 # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)
        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)
        # validate
        charErrorRate = validate(model, loader)
        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            # Use corrected path for saving accuracy
            accuracy_file_path = os.path.join(FilePaths.fnModel, 'accuracy.txt')
            try:
                with open(accuracy_file_path, 'w') as f:
                    f.write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
            except Exception as e:
                 print(f"Warning: Could not write accuracy file to {accuracy_file_path}: {e}")
        else:
            print('Character error rate not improved')
            noImprovementSince += 1
        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    "validate NN"
    # (Your existing validate code - unchanged)
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)
        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
           dist = Levenshtein.distance(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    return charErrorRate
# --- End Training/Validation Functions ---


# --- Inference Functions ---
def infer(model, fnImg):
    """
    Recognize text in image provided by file path (for command-line use).
    Uses the globally loaded model instance.
    """
    global model_instance
    if model_instance is None:
        print("Error: Model not loaded. Cannot infer.")
        return

    try:
        print(f"Inferring image (cmd line): {fnImg}")
        img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), model_instance.imgSize)
        if img is None:
            print(f"Error: Could not read or preprocess image at {fnImg}")
            return
        batch = Batch(None, [img]) # Create a batch with a single image
        (recognized, probability) = model_instance.inferBatch(batch, True) # Use global model
        if recognized and probability:
            # Remove spaces as per original code
            recognized_text = recognized[0].replace(" ", "")
            print('Recognized:', '"' + recognized_text + '"')
            print('Probability:', probability[0])
        else:
            print("Inference did not return recognized text or probability.")
    except Exception as e:
        print(f"Error during inference for {fnImg}: {e}")
        import traceback
        print(traceback.format_exc())


def infer_by_web(path):
    """
    Recognize text in an image for the web application.
    Uses the globally loaded model and character list.

    Args:
        path (str): Full path to the uploaded image file.

    Returns:
        tuple: (recognized_text, probability) or ("Error: Model not loaded", 0.0)
               or ("Error during prediction", 0.0) on failure.
    """
    global model_instance, character_list
    print(f"[main.py - infer_by_web] Processing image: {path}")

    if model_instance is None or character_list is None:
        print("[main.py - infer_by_web] Error: Global model or char list not loaded.")
        return "Error: Model not loaded", 0.0

    try:
        # Preprocess the image using the size defined in the Model class
        img = preprocess(cv2.imread(path, cv2.IMREAD_GRAYSCALE), model_instance.imgSize)
        if img is None:
            print(f"[main.py - infer_by_web] Error: Failed to read/preprocess image at {path}")
            return "Error: Image loading failed", 0.0

        # Create a batch containing the single image
        batch = Batch(None, [img])

        # Perform inference using the globally loaded model
        (recognized, probability) = model_instance.inferBatch(batch, True)

        # Process and return results
        if recognized and probability:
             # Remove spaces as per original logic
            recognized_text = recognized[0].replace(" ", "")
            prob = probability[0]
            print(f'[main.py - infer_by_web] Recognized: "{recognized_text}", Probability: {prob}')
            return recognized_text, float(prob) # Ensure probability is float
        else:
             print("[main.py - infer_by_web] Inference returned empty result.")
             return "Error: Inference failed", 0.0

    except Exception as e:
        print(f"[main.py - infer_by_web] Error during web inference for {path}: {e}")
        import traceback
        print(traceback.format_exc())
        return "Error during prediction", 0.0
# --- End Inference Functions ---


# --- Main Execution Block (for command-line use) ---
def main_cli():
    "main function for command line execution"
    # optional command line args
    parser = argparse.ArgumentParser(description="Train or infer on Devanagari OCR model.")
    parser.add_argument("--train", help="Train the NN", action="store_true")
    parser.add_argument("--validate", help="Validate the NN", action="store_true")
    parser.add_argument("--beamsearch", help="Use beam search decoding", action="store_true")
    parser.add_argument("--wordbeamsearch", help="Use word beam search decoding", action="store_true")
    parser.add_argument("--inferfile", help="Run inference on a specific file", type=str, default=FilePaths.fnInfer)
    args = parser.parse_args()

    # Note: For CLI, model loading happens here, separate from global load for web
    current_model = None
    current_char_list = None

    # Determine decoder type for CLI execution
    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # Load Character List for CLI
    try:
        with codecs.open(FilePaths.fnCharList, encoding='utf-8') as f:
            current_char_list = f.read()
    except Exception as e:
        print(f"CLI Error: Failed to load charList {FilePaths.fnCharList}: {e}")
        sys.exit(1)

    # train or validate on dataset
    if args.train or args.validate:
        # load training data, create TF model
        try:
             # Ensure paths are correct for DataLoader when run from CLI
            loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
            print("DataLoader initialized successfully for Train/Validate.")
        except Exception as e:
             print(f"CLI Error: Failed to initialize DataLoader with path {FilePaths.fnTrain}: {e}")
             sys.exit(1)


        # save characters of model for inference mode (needed if training creates a new char list)
        if args.train:
            try:
                 # Use corrected path
                charlist_path = os.path.join(FilePaths.fnModel, 'charList.txt')
                with open(charlist_path, 'w', encoding='UTF-8') as f:
                     f.write(str().join(loader.charList))
                current_char_list = loader.charList # Use the newly created list
                print(f"Saved new character list to {charlist_path}")
            except Exception as e:
                 print(f"CLI Warning: Could not save charList to {charlist_path}: {e}")


        # save words contained in dataset into file
        try:
             # Use corrected path
             corpus_path = os.path.join(FilePaths.fnModel, 'corpus.txt') # Save corpus in model dir
             with open(corpus_path, 'w', encoding='UTF-8') as f:
                 f.write(str(' ').join(loader.trainWords + loader.validationWords))
             print(f"Saved corpus file to {corpus_path}")
        except Exception as e:
             print(f"CLI Warning: Could not save corpus file to {corpus_path}: {e}")

        # execute training or validation
        if args.train:
            # *** CLI: Remove model_path_base here too for consistency if needed ***
            current_model = Model(current_char_list, decoderType)
            train(current_model, loader)
        elif args.validate:
            # Typically load a pre-trained model for validation
            # *** CLI: Remove model_path_base here too for consistency if needed ***
            current_model = Model(current_char_list, decoderType, mustRestore=True)
            validate(current_model, loader)

    # infer text on test image (CLI mode)
    else:
        print("--- Running Inference (Command Line) ---")
        # Display accuracy from file if it exists
        try:
            accuracy_file = os.path.join(FilePaths.fnModel, 'accuracy.txt')
            if os.path.exists(accuracy_file):
                 print(f"Accuracy info from {accuracy_file}:")
                 print(open(accuracy_file).read())
            else:
                 print(f"Accuracy file not found at {accuracy_file}")
        except Exception as e:
            print(f"Could not read accuracy file: {e}")

        # Load model specifically for CLI inference
        print(f"Loading model for CLI inference...")
        # *** CLI: Remove model_path_base here too for consistency if needed ***
        current_model = Model(current_char_list, decoderType, mustRestore=True)
        infer(current_model, args.inferfile) # Use the locally loaded model

if __name__ == '__main__':
    main_cli() # Execute command-line logic only when run directly
# --- End Main Execution Block ---
