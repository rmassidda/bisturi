#! bash

log () {
  d_print="$(date +"%Y-%m-%d %H:%M:%S")"
  echo "[$d_print]" "$@"
}

check () {
  md5="$(md5sum "$1" | cut -d ' ' -f1)"
  if [[ x"$md5" == x"$2" ]]; then
    log "Matching MD5 $1 $2"
  else
    log "Wrong MD5 $1 $2 $md5"
    exit 1
  fi
}

# Default values
NW=4

# Parse arguments
while getopts "h:w:" opt; do
  case "$opt" in
    h)
      echo "Usage: `basename $0` [-h] [-w num_workers] base_directory"
      echo "  -h: Displays this help message"
      echo "  -w: Number of workers for parallel data"
      echo "      preprocessing. (Default 4)"
      exit 0
      ;;
    w)
      NW=$OPTARG
      ;;
    /?)
      echo "Invalid option -$OPTARG" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND-1))

this="$(pwd)"

log "Installing our library"
pip3 install ./bisturi/ | grep -v "Requirement already satisfied"

log "Instaling other dependencies"
pip3 install ipywidgets tabulate matplotlib xmltodict | grep -v "Requirement already satisfied"

log "Init base directiory in $1 from $this"
mkdir -p "$1" 
mkdir -p "$1/models"
mkdir -p "$1/datasets"
mkdir -p "$1/results"

log "Copying temporary files into the base directory"
cp -r "various" "$1"

log "Entering base directory $1"
cd "$1"

log "Downloading pre-trained models"
cd "models"
wget -nc http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar
wget -nc http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar
wget -nc http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar

log "Checking MD5 sum"
check alexnet_places365.pth.tar c9c7e52228ef406ca64f8189585e2a28
check resnet18_places365.pth.tar 7a8808eefd4ea7458650ca84855f69b1
check densenet161_places365.pth.tar 07efb6c7f25f832cb567d3644a131e6a

log "Entering directory $1/datasets"
cd ../datasets

log "Downloading Broden dataset"
wget -nc http://netdissect.csail.mit.edu/data/broden1_224.zip

log "Checking MD5 sum"
check broden1_224.zip be13c292eaef3300da03f8d0e48eab1c

log "Unzip Broden"
test -d broden1_224 || unzip -q broden1_224.zip

log "Copy original ontological annotation"
mv ../various/broden_wordnet_alignment.csv broden1_224/
mv ../various/ontology.txt broden1_224/
mv ../various/reverse_index.json  broden1_224/

log "Downloading ILSVRC2011"
mkdir "ilsvrc2011"
cd "ilsvrc2011"
wget -nc --no-check-certificate "https://image-net.org/data/ILSVRC/2011/ILSVRC2011_bbox_val.v3.tar"
wget -nc --no-check-certificate "https://image-net.org/data/ILSVRC/2011/ILSVRC2011_images_val.tar"

log "Checking MD5 sum"
check ILSVRC2011_bbox_val.v3.tar e1616eddd965c2b4709fa393fa791b3c
check ILSVRC2011_images_val.tar e2026d95f07299a62125a3c264e3adf1

log "Unpacking ILSVRC2011"
if ! [[ -d val ]]; then
      tar -xf "ILSVRC2011_bbox_val.v3.tar"
      tar -xf "ILSVRC2011_images_val.tar"
fi

log "Preprocessing ImageNet with $NW processes"
if ! [[ -d out ]]; then
  mkdir out
  python3 "$this/scripts/preprocess_imagenet.py" --size 224 --strategy "center" --nw "$NW" val out || rmdir out
fi

log "Copying ontology annotation (ImageNet)"
cp ../../various/imagenet_ontology.txt out/ontology.txt
cp ../../various/imagenet_reverse_index.json out/reverse_index.json

log "Downloading WordNet"
python3 "$this/scripts/wordnet.py"
