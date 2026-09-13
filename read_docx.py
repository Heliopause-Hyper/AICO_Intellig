import zipfile
import xml.etree.ElementTree as ET

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.XML(xml_content)
            namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text = []
            for paragraph in tree.iterfind('.//w:p', namespace):
                texts = [node.text for node in paragraph.iterfind('.//w:t', namespace) if node.text]
                if texts:
                    text.append(''.join(texts))
            return '\n'.join(text)
    except Exception as e:
        return str(e)

print("=== File 1: 0投稿分析1.docx ===")
print(extract_text_from_docx('/home/ycl/AICO1/0投稿分析1.docx'))
print("\n=== File 2: 0投稿分析1-Advances in Engineering Software.docx ===")
print(extract_text_from_docx('/home/ycl/AICO1/0投稿分析1-Advances in Engineering Software.docx'))
