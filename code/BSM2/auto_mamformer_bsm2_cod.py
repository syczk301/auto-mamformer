import sys

from auto_mamformer_bsm2 import main


if __name__ == "__main__":
    main(["--target", "cod", *sys.argv[1:]])
