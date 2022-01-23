from PIL import Image
from glob import glob
from multiprocessing import Pool
import argparse
import json
import os
import sys
import xmltodict

def pad_image(img, bg_color=0, boxes=[]):
    '''
    Given a rectangular image it
    returns a square one, padded
    with a given background color.
    '''
    width, height = img.size
    if width == height:
        return img, boxes

    max_size = max(width, height)
    result = Image.new(img.mode, (max_size, max_size), bg_color)

    offset = (0, (width - height) // 2) if width > height else ((height - width) // 2, 0)
    result.paste(img, offset)
    return result, [tuple(point_move(p, *offset) for p in box) for box in boxes]

def crop_image(img, center=None, boxes=[]):
    '''
    Given a rectangular image
    it returns a square one,
    obtained by cropping in
    the 
    '''
    width, height = img.size
    if width == height:
        return img, boxes

    # Default to the center of the image
    if center is None:
        center = Point(width//2, height//2)

    min_size = min(width, height)

    if width > height:
        offset = max(0, center.x - (height//2)) # Left border
        offset = min(offset, width - height)    # Right border
        offset = (offset, 0)
    else:
        offset = max(0, center.y - (width//2)) # Top border
        offset = min(offset, height - width)   # Bottom border
        offset = (0, offset)
    result = img.crop((*offset, offset[0] + min_size, offset[1] + min_size))

    return result, [tuple(point_move(p, *offset, direction=-1) for p in box) for box in boxes]

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x,self.y))

def point_to_tuple(point):
    return (point.x, point.y)

def point_resize(point, width, height):
    point.x = point.x // width
    point.y = point.y // height
    return point

def point_move(point, dx, dy, direction=1):
    point.x += (dx * direction)
    point.y += (dy * direction)
    return point

def _parse_bbox(entry):
    xmin = int(entry['bndbox']['xmin'])
    xmax = int(entry['bndbox']['xmax'])
    ymin = int(entry['bndbox']['ymin'])
    ymax = int(entry['bndbox']['ymax'])

    return (Point(xmin, ymin), Point(xmax, ymax))

def parse_bbox(xml_dict):
    '''
    Given an annotation from
    the XML data, it returns
    a list of all the bounding
    boxes.
    '''
    obj = xml_dict['annotation']['object']
    if isinstance(obj, list):
        return [(e['name'],_parse_bbox(e)) for e in obj]
    else:
        return [(obj['name'],_parse_bbox(obj))]

def worker(worker_id, image_files, strategy, new_size, out_dir):
    '''
    Given a set of images it
    stores the preprocessed
    images and returns the
    partial index.
    '''
    errors = []
    images_index = []
    n_images = len(image_files)

    i = 0
    for image_path in image_files:
        # Various operations over the filename
        image_dir  = os.path.dirname(image_path)
        image_base = os.path.basename(image_path)
        image_name = '.'.join(image_base.split('.')[:-1]) 
        xml_path   = os.path.join(image_dir, image_name + '.xml')
        out_path   = os.path.join(out_dir, image_name + '_' + str(new_size) + '_' + strategy + '.JPEG')

        # Load from file
        im = Image.open(image_path)
        with open(xml_path) as fp:
            xml_dict = xmltodict.parse(fp.read())

        # Retrieve size from the XML
        xml_size = xml_dict['annotation']['size']
        xml_size = (int(xml_size['width']), int(xml_size['height']))

        # Check if the annotation matches the image size
        if im.size != xml_size:
            errors.append((i, image_path))
            continue

        # Parse bounding boxes
        # It produces a list
        # [(synset,(pmin, pmax))]
        annotations = parse_bbox(xml_dict)
        synsets = [e[0] for e in annotations]
        boxes   = [e[1] for e in annotations]

        # Eventually square the image
        if strategy == 'pad':
            im, boxes = pad_image(im, boxes=boxes)
        elif strategy == 'center':
            im, boxes = crop_image(im, boxes=boxes)
        elif strategy == 'box':
            pmin, pmax = boxes[0]
            center     = Point((pmin.x+pmax.x)//2, (pmin.y+pmax.y)//2)
            im, boxes  = crop_image(im, center=center, boxes=boxes)
        elif strategy == 'resize' or strategy == 'original':
            pass

        if strategy != 'original':
            # Resize
            w_ratio = im.size[0] / new_size
            h_ratio = im.size[1] / new_size
            boxes = [tuple(point_resize(p, w_ratio, h_ratio) for p in box) for box in boxes]
            im = im.resize((new_size,new_size))

            # Store
            im.save(out_path)

            # Update the index
            images_index.append({
                'width': new_size,
                'height': new_size,
                'path': os.path.basename(out_path),
                'boxes': [(synset, tuple(point_to_tuple(p) for p in box)) for synset, box in zip(synsets,boxes)]
            })
        else:
            images_index.append({
                'width': xml_size[0],
                'height': xml_size[1],
                'path': image_path,
                'boxes': [(synset, tuple(point_to_tuple(p) for p in box)) for synset, box in zip(synsets,boxes)]
            })

        if i % 1000 == 0: print('Worker', worker_id, '%.2f' % (100*i/n_images) + '%')
        i += 1

    return images_index, errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for the annotated ImageNet dataset')
    parser.add_argument('dataset_path')
    parser.add_argument('out_dir')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--strategy', type=str, default='center')
    parser.add_argument('--nw', type=int, default=0)
    args = parser.parse_args()

    dset      = args.dataset_path
    out_dir   = args.out_dir
    new_size  = args.size
    strategy  = args.strategy
    n_workers = args.nw

    # check if the strategy is admitted
    if strategy not in ['pad','center','box','resize', 'original']:
        print('Unknown strategy', file=sys.stderr)
        print('Admitted strategies are: pad, center, box, resize, original', file=sys.stderr)
        sys.exit(1)

    # select all the JPEG files in the dataset
    image_files = list(glob(os.path.join(dset, '*.JPEG')))
    n_images    = len(image_files)
    print('Ready to analyze over', n_images)

    # path of the JSON index
    if strategy != 'original':
        idx_path = os.path.join(out_dir, '_'.join(['index', str(new_size), strategy]) + '.json')
    else:
        idx_path = os.path.join(out_dir, 'index.json')

    # parallel computation
    if n_workers > 0:
        pool = Pool(n_workers)

        # batch size
        bs = int(n_images / n_workers)
        bs = bs if bs != 0 else n_images
        ranges = [(start_idx, min(n_images, start_idx+bs)) for start_idx in range(0, n_images, bs)]
        params = [(i, image_files[a:b], strategy, new_size, out_dir) for i,(a,b) in enumerate(ranges)]

        # Map
        partial = pool.starmap(worker, params)

        # Reduce
        images_index = sum([e[0] for e in partial], [])
        errors       = sum([e[1] for e in partial], [])

    else:
        images_index, errors = worker(0, image_files, strategy, new_size, out_dir)

    print(len(images_index), 'images done')
    print(len(errors), 'images were not analyzed')

    # Indices images
    images_index = [{'idx': i, **images_index[i]} for i, _ in enumerate(images_index)]

    # Dump index for preprocessed images
    with open(idx_path, 'w+') as fp:
        json.dump(images_index, fp, indent=2)

    print('Results stored in', idx_path)
