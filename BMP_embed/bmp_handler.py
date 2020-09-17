import struct, array
import numpy as np


class Handler:
    def __init__(self, path: str):
        # 存储图片路径，方便后续使用
        self.path = path

        # 以二进制只读方式打开图片
        with open(self.path, "rb") as f:
            # 读取文件头并按照BMP协议对文件头进行字节解析
            header = struct.unpack("<2sI2H4I2H6I", f.read(0x36))
            # 将文件头中的信息存储到实例中
            self.rgb_offset = header[4]  # RGB偏移量(byte)

    # 将字符串string嵌入当前图片，并将结果存储到dst
    def embed(self, string: str, dst: str):
        # 计算需要修改的像素长度
        pixel_len = 3 * (len(string) + 1)

        # 读取原图片
        with open(self.path, "rb") as f:
            # 暂存头部信息
            former = f.read(self.rgb_offset)
            # 读取一定长度的RGB并转化为二维矩阵
            rgb_bytes = f.read(pixel_len)
            rgb_list = array.array('B', rgb_bytes).tolist()
            rgb_matrix = np.reshape(np.array(rgb_list), (-1, 3))
            # 暂存结尾信息
            latter = f.read()

        # 将字符串转换为ASCII数组
        embed_code = list()
        for char in string:
            embed_code.append(ord(char))

        # 将图片前N个像素的R值替换为字符的ASCII码
        for i in range(len(embed_code)):
            rgb_matrix[i][2] = embed_code[i]
        # 设置终止符
        rgb_matrix[len(embed_code)][2] = 0

        # 存储新图片
        with open(dst, "wb") as f:
            # 将RGB矩阵转换回二进制序列
            new_rgb_list = rgb_matrix.reshape(-1).tolist()
            new_rgb_bytes = bytes()
            for i in new_rgb_list:
                new_rgb_bytes += i.to_bytes(length=1, byteorder='big')
            # 拼接信息形成完整图片
            new_img = former + new_rgb_bytes + latter
            # 写入新图片
            f.write(new_img)

    # 从当前图片解密字符串
    def extract(self):
        with open(self.path, "rb") as f:
            f.seek(self.rgb_offset)
            # 读取RGB并转化为二维矩阵
            rgb_bytes = f.read()
            rgb_list = array.array('B', rgb_bytes).tolist()
            rgb_matrix = np.reshape(np.array(rgb_list), (-1, 3))

        # 读取ASCII码直到遇到终止符
        embed_code = list()
        for i in rgb_matrix:
            if i[2] != 0:
                embed_code.append(i[2])
            else:
                break

        # 返回ASCII序列所对应的字符串
        result = str()
        for i in embed_code:
            result += chr(i)
        return result


def Embed(src: str, string: str, dst: str):
    handler = Handler(src)
    handler.embed(string, dst)


def Extract(src: str):
    handler = Handler(src)
    print(handler.extract())


if __name__ == '__main__':
    Embed("image.bmp", "This is a test.", "new_img.bmp")
    Extract("new_img.bmp")
