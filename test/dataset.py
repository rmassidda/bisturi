from argparse import ArgumentParser
from enum import Enum, auto
from tqdm.auto import tqdm
import sys
import time

from bisturi.dataset.broden import BrodenDataset
from bisturi.dataset.broden import BrodenOntology
from bisturi.dataset.imagenet import ImageNetDataset
from bisturi.dataset.imagenet import ImageNetOntology
from bisturi.dataset import concept_mask


class DatasetTypes(Enum):
    Broden = auto(),
    ImageNet = auto()


def main(dset_path: str, dset_type: DatasetTypes):
    if dset_type == DatasetTypes.Broden:
        ontology = BrodenOntology(dset_path)
        dataset = BrodenDataset(dset_path)
    elif dset_type == DatasetTypes.ImageNet:
        ontology = ImageNetOntology(dset_path)
        dataset = ImageNetDataset(dset_path)

    concepts = ontology.to_list()

    print(f'{len(concepts)} concepts')
    print(f'{len(dataset)} images')

    time_extract = 0.
    time_cmask = 0.

    pbar = tqdm(range(len(dataset)), total=len(dataset))

    for idx in pbar:
        time_a = time.time()
        img, segmentation = dataset[idx]
        time_b = time.time()
        time_extract = time_b - time_a

        time_c = time.time()
        for concept in concepts:
            _ = concept_mask(segmentation, concept)
        time_d = time.time()
        time_cmask = time_d - time_c

        pbar.set_description(
            f'{time_extract:.3f} x img, '
            f'{time_cmask:.3f} x cmask '
            f'{segmentation.shape} '
            f'{segmentation.dtype} '
            f'{type(segmentation)}'
        )


if __name__ == '__main__':
    # Argument parsing
    # -b: flag to use Broden
    # -i: flag to use ImageNet
    # dset_path: path to dataset
    parser = ArgumentParser()
    parser.add_argument('-b', '--broden', action='store_true',
                        help='Use Broden dataset')
    parser.add_argument('-i', '--image-net', action='store_true',
                        help='Use ImageNet dataset')
    parser.add_argument('dset_path', type=str, help='Path to dataset')
    args = parser.parse_args()

    # Check which type of dataset
    if args.broden:
        dset_type = DatasetTypes.Broden
    elif args.image_net:
        dset_type = DatasetTypes.ImageNet
    else:
        print('Please specify dataset type', file=sys.stderr)
        sys.exit(1)

    # Run main
    main(args.dset_path, dset_type)
