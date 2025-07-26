import json
import uuid

class SampleWriter:
    """
    将生成的样本数据写入文件。
    """
    def __init__(self, output_path: str):
        self.output_path = output_path
        open(self.output_path, 'w').close()

    def write_sample(self, data: dict):
        """
        写入单个样本数据。

        :param data: 样本数据字典
        """
        data['id'] = str(uuid.uuid4())
        with open(self.output_path, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')