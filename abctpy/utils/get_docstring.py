import os


def from_matlab(filename):

    # read contents of matlab function
    base_dir = os.path.dirname(__file__)
    filename = os.path.join(base_dir, "..", "..", "..", "matlab", filename)
    with open(filename, "r") as f:
        filetext = f.readlines()

    # parse the filetext to get the docstring
    for i, line in enumerate(filetext):
        if not line.strip().startswith("%"):
            if i > 0:
                break

    docstring = "".join([line[1:] for line in filetext[1:i]])
    docstring = docstring.replace("[", "").replace("]", "")
    return docstring
