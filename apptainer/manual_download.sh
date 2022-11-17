#!/bin/bash
APPTAINER_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
PROJECT="$(dirname "$APPTAINER_ROOT")"

FMNIST_TRAIN_X="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
FMNIST_TRAIN_Y="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
FMNIST_TEST_X="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
FMNIST_TEST_Y="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
CIFAR_10="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_100="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
MED_MNIST_PATH="https://zenodo.org/record/6496656/files/pathmnist.npz"
MED_MNIST_RET="https://zenodo.org/record/6496656/files/retinamnist.npz"
MED_MNIST_OCT="https://zenodo.org/record/6496656/files/octmnist.npz"


FMNIST_OUTDIR="$PROJECT/data/fmnist/FashionMNIST/raw"
CIFAR_10_OUTDIR="$PROJECT/data/cifar10"
CIFAR_100_OUTDIR="$PROJECT/data/cifar100"
MED_MNIST_OUTDIR="$PROJECT/data/medmnist"
MED_MNIST_PATH_OUTDIR="$MED_MNIST_OUTDIR/PathMNIST"
MED_MNIST_RET_OUTDIR="$MED_MNIST_OUTDIR/RetinaMNIST"
MED_MNIST_OCT_OUTDIR="$MED_MNIST_OUTDIR/OCTMNIST"

FMNIST_TRAIN_X_OUTFILE="$FMNIST_OUTDIR/train-images-idx3-ubyte.gz"
FMNIST_TRAIN_Y_OUTFILE="$FMNIST_OUTDIR/train-labels-idx1-ubyte.gz"
FMNIST_TEST_X_OUTFILE="$FMNIST_OUTDIR/t10k-images-idx3-ubyte.gz"
FMNIST_TEST_Y_OUTFILE="$FMNIST_OUTDIR/t10k-labels-idx1-ubyte.gz"
CIFAR_10_OUTFILE="$CIFAR_10_OUTDIR/cifar-10-python.tar.gz"
CIFAR_100_OUTFILE="$CIFAR_100_OUTDIR/cifar-100-python.tar.gz"
MED_MNIST_PATH_OUTFILE="$MED_MNIST_PATH_OUTDIR/pathmnist.npz"
MED_MNIST_RET_OUTFILE="$MED_MNIST_RET_OUTDIR/retinamnist.npz"
MED_MNIST_OCT_OUTFILE="$MED_MNIST_OCT_OUTDIR/octmnist.npz"

EFFICIENTNET_V2_S="https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"
WIDE_RESNET50_2="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth"
EFFICIENTNET_V2_S_OUTDIR="$PROJECT/torchmodels/hub/checkpoints"
WIDE_RESNET50_2_OUTDIR="$PROJECT/torchmodels/hub/checkpoints"
EFFICIENTNET_V2_S_OUTFILE="$EFFICIENTNET_V2_S_OUTDIR/efficientnet_v2_s-dd5fe13b.pth"
WIDE_RESNET50_2_OUTFILE="$WIDE_RESNET50_2_OUTDIR/wide_resnet50_2-9ba9bcbe.pth"

FMNIST_TRAIN_X_CHECKFILE="$FMNIST_OUTDIR/train-images-idx3-ubyte"
FMNIST_TRAIN_Y_CHECKFILE="$FMNIST_OUTDIR/train-labels-idx1-ubyte"
FMNIST_TEST_X_CHECKFILE="$FMNIST_OUTDIR/t10k-images-idx3-ubyte"
FMNIST_TEST_Y_CHECKFILE="$FMNIST_OUTDIR/t10k-labels-idx1-ubyte"
CIFAR_10_CHECKFILE="$CIFAR_10_OUTDIR/cifar-10-batches-py"
CIFAR_100_CHECKFILE="$CIFAR_100_OUTDIR/cifar-100-python"
EFFICIENTNET_V2_S_CHECKFILE="$EFFICIENTNET_V2_S_OUTDIR/efficientnet_v2_s-dd5fe13b.pth"
WIDE_RESNET50_2_CHECKFILE="$WIDE_RESNET50_2_OUTDIR/wide_resnet50_2-9ba9bcbe.pth"


download () {
    local URL="$1"
    local LOC="$2"
    local CHK="$3"
    if [ -d "$CHK" ] || [ -f "$CHK" ]; then
        echo "$CHK already exists. Skipping download."
        return 0
    fi
    mkdir -p "$(dirname "$LOC")"
    wget "$URL" -O "$LOC"
}

download_med_mnist () {
    local URL="$1"
    local LOC="$2"
    local CHK="$3"
    if [ -f "$CHK" ]; then
        echo "$CHK already exists. Skipping download."
        return 0
    fi
    mkdir -p "$(dirname "$LOC")"
    mkdir -p "$LOC"
    cd "$LOC" || exit
    wget "$URL"
    cd "$PROJECT" || exit
}

untar () {
    local LOC="$1"
    local OUT="$2"
    cd "$LOC" || exit
    if [ ! -f "$OUT" ]; then
        echo "No archive at $OUT to extract. Skipping."
        return 0
    fi
    echo "Decompressing $LOC..."
    tar -xvf "$OUT"
    rm -rf "$OUT"
}

ungzip () {
    local LOC="$1"
    local OUT="$2"
    cd "$LOC" || exit
    if [ ! -f "$OUT" ]; then
        echo "No archive at $OUT to extract. Skipping."
        return 0
    fi
    echo "Decompressing $LOC..."
    gzip -df "$OUT"
}

cd "$PROJECT" || exit
download "$FMNIST_TRAIN_X" "$FMNIST_TRAIN_X_OUTFILE" "$FMNIST_TRAIN_X_CHECKFILE"
download "$FMNIST_TRAIN_Y" "$FMNIST_TRAIN_Y_OUTFILE" "$FMNIST_TRAIN_Y_CHECKFILE"
download "$FMNIST_TEST_X" "$FMNIST_TEST_X_OUTFILE" "$FMNIST_TEST_X_CHECKFILE"
download "$FMNIST_TEST_Y" "$FMNIST_TEST_Y_OUTFILE" "$FMNIST_TEST_Y_CHECKFILE"
download "$CIFAR_10" "$CIFAR_10_OUTFILE" "$CIFAR_10_CHECKFILE"
download "$CIFAR_100" "$CIFAR_100_OUTFILE" "$CIFAR_100_CHECKFILE"
download "$EFFICIENTNET_V2_S" "$EFFICIENTNET_V2_S_OUTFILE" "$EFFICIENTNET_V2_S_CHECKFILE"
download "$WIDE_RESNET50_2" "$WIDE_RESNET50_2_OUTFILE" "$WIDE_RESNET50_2_CHECKFILE"
download_med_mnist "$MED_MNIST_RET" "$MED_MNIST_RET_OUTDIR" "$MED_MNIST_RET_OUTFILE"
download_med_mnist "$MED_MNIST_PATH" "$MED_MNIST_PATH_OUTDIR" "$MED_MNIST_PATH_OUTFILE"
download_med_mnist "$MED_MNIST_OCT" "$MED_MNIST_OCT_OUTDIR" "$MED_MNIST_OCT_OUTFILE"

untar "$CIFAR_10_OUTDIR" "$CIFAR_10_OUTFILE"
untar "$CIFAR_100_OUTDIR" "$CIFAR_100_OUTFILE"
ungzip "$FMNIST_OUTDIR" "$FMNIST_TRAIN_X_OUTFILE"
ungzip "$FMNIST_OUTDIR" "$FMNIST_TRAIN_Y_OUTFILE"
ungzip "$FMNIST_OUTDIR" "$FMNIST_TEST_X_OUTFILE"
ungzip "$FMNIST_OUTDIR" "$FMNIST_TEST_Y_OUTFILE"
