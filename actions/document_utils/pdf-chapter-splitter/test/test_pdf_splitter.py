import PyPDF2
import os

def get_pdf_outline_info(outline, reader, parent_title=""):
    """
    递归地从PDF大纲中提取所有书签的标题和页码。
    PyPDF2的页码是基于0的索引。
    """
    outline_info = []
    for item in outline:
        if isinstance(item, PyPDF2.generic.Destination):
            title = item.title
            # PyPDF2.generic.Destination.page 是一个 IndirectObject，需要解析
            # 获取页码索引，然后通过 reader.get_page_number(page_object) 获取实际页码
            page_index = reader.get_page_number(item.page)
            outline_info.append({"title": title, "page_index": page_index})
        elif isinstance(item, list):
            # 处理嵌套书签
            outline_info.extend(get_pdf_outline_info(item, reader, parent_title))
    return outline_info

def calculate_page_ranges(outline_info, total_pages):
    """
    根据书签信息和总页数计算每个书签对应的页码范围。
    返回的页码是基于1的索引。
    """
    sections = []
    
    # 按页码排序书签，确保顺序正确
    outline_info.sort(key=lambda x: x["page_index"])

    for i, item in enumerate(outline_info):
        title = item["title"]
        start_index = item["page_index"]

        end_index = total_pages - 1 # 默认结束页为文档最后一页（0-based）
        if i + 1 < len(outline_info):
            end_index = outline_info[i+1]["page_index"] - 1 # 下一个书签的起始页码前一页

        # 确保结束页码不小于起始页码
        if start_index <= end_index:
            sections.append({
                "name": title,
                "start_page": start_index + 1, # 转换为1-based
                "end_page": end_index + 1    # 转换为1-based
            })
        elif i == len(outline_info) - 1: # 如果是最后一个书签，且start_index > end_index (可能因为只有一个书签或书签指向最后一页)
            sections.append({
                "name": title,
                "start_page": start_index + 1,
                "end_page": total_pages
            })

    return sections

def run_chapter_recognition_test(pdf_path):
    """
    主测试函数，用于识别PDF章节并模拟输出。
    """
    if not os.path.exists(pdf_path):
        print(f"错误：文件 '{pdf_path}' 不存在。请检查路径是否正确。")
        return

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            print(f"PDF文件总页数：{total_pages}")

            outline = reader.outline
            if not outline:
                print("PDF文件中没有找到书签（大纲）信息。")
                # 如果没有书签，可以考虑回退到文本模式匹配，但目前只测试书签模式
                return

            outline_info = get_pdf_outline_info(outline, reader)
            
            # 过滤掉页码为None的书签（可能指向外部链接等）
            outline_info = [item for item in outline_info if item["page_index"] is not None]

            if not outline_info:
                print("未找到有效的书签信息。")
                return

            page_ranges = calculate_page_ranges(outline_info, total_pages)

            print("\n分离后的文件名及原始页码起始与结束位置：")
            for section in page_ranges:
                print(f"文件名: {section['name'].replace(' ', '_').replace(':', '')}.pdf, 原始页码: {section['start_page']}-{section['end_page']}")

    except Exception as e:
        print(f"处理PDF文件时发生错误：{e}")

if __name__ == "__main__":
    # 请将此路径替换为您的实际PDF文件路径
    pdf_file_path = r"C:\Users\Jorkey\Downloads\Dream Yoga Illuminating Your Life Through Lucid Dreaming and the Tibetan Yogas of Sleep (Andrew Holecek) (Z-Library).pdf"
    run_chapter_recognition_test(pdf_file_path)