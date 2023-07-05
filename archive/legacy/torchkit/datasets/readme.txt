The `datasets` API contains several datasets and datamodules of commonly used
open datasets.

The datasets can be categorized into 2 types:
    - Single-task dataset: each set of images are only associated with a single
      set of ground-truth. The dataset can have several types of ground-truth
      labels, but they are not related to each other.

    - Multi-tasks dataset: this type of dataset only has one set of images, but
      there are several ground-truth label types associated with it. For
      example: the COCO dataset supports bbox, polygon, caption, and person
      keypoints.
