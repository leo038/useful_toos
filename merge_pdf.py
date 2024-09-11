from PyPDF2 import PdfWriter, PdfReader
from PIL import Image
import img2pdf
import os

def merge_pdfs(file_list, output_file):
    pdf_writer = PdfWriter()

    for pdf_file in file_list:
        pdf_reader = PdfReader(pdf_file)
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])

    with open(output_file, 'wb') as out:
        pdf_writer.write(out)
    print(f"合成后保存的pdf文件：{output_file}")


def convert_img(image_path):
    # # 打开图片
    # img = Image.open(image_path)
    #
    # # 将图片转换为JPEG格式的字节流
    # output = io.BytesIO()
    # img.convert('RGB').save(output, format='JPEG')
    # data = output.getvalue()
    #
    # # 创建一个PDF写入对象
    # writer = PdfWriter()
    #
    # # 将图片作为一个页面添加到PDF
    # page = writer.add_page()
    #
    # # 这里的dpi值可以根据需要调整
    # page.add_image(data, dpi=150)
    #
    #
    # save_path = image_path.split(".")[0] + ".pdf"
    #
    # # 写入PDF文件
    # with open(save_path, 'wb') as out:
    #     writer.write(out)

    # using img2pdf library
    # importing necessary libraries


    # opening image
    # image = Image.open(image_path)
    # # converting into chunks using img2pdf
    # pdf_bytes = img2pdf.convert(image.filename)
    # # opening or creating pdf file
    #
    # pdf_path = image_path.split(".")[0] + ".pdf"
    # file = open(pdf_path, "wb")
    # # writing pdf files with chunks
    # file.write(pdf_bytes)
    # # closing image file
    # image.close()
    # # closing pdf file
    # file.close()
    # # output
    # print("Successfully made pdf file")
    pdf_path = image_path.replace(".jpg", ".pdf")
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(image_path,rotation=img2pdf.Rotation.ifvalid))

    print(f"保存的pdf文件： {pdf_path}")

    print("Successfully made pdf file")

if __name__ == "__main__":

    base_dir = "./data"
    folder_list = os.listdir(base_dir)
    for folder in folder_list:
        print(f"正在处理{folder}")
        file_list = os.listdir(os.path.join(base_dir, folder))
        file_list.sort()
        print(f"文件列表： {file_list}")
        convert_file_list = []
        for file in file_list:
            if not file.endswith('.pdf'):
                print(f"文件{file}不是pdf文件， 需要进行转换")
                image_path = os.path.join(base_dir, folder, file)
                convert_img(image_path)
            new_name =os.path.join(base_dir, folder, file.split(".")[0] + ".pdf")
            convert_file_list.append(new_name)


        save_name = os.path.join(base_dir, folder + ".pdf")
        merge_pdfs(convert_file_list, save_name)