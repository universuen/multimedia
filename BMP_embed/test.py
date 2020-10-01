from .bmp_handler import Handler


def embed(src: str, string: str, dst: str):
    handler = Handler(src)
    handler.embed(string, dst)


def extract(src: str):
    handler = Handler(src)
    print(handler.extract())


if __name__ == '__main__':
    embed("image.bmp", "This is a test!", "new_img.bmp")
    extract("new_img.bmp")
