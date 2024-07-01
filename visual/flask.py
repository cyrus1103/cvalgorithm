from flask import Flask, request, jsonify
app = Flask(__name__)


@app.route('/infer', methods=['POST'])
def infer():
    # 这里添加调用模型进行推理的代码
    # 假设使用ONNX Runtime为例
    import onnxruntime
    session = onnxruntime.InferenceSession('path_to_your_model.onnx')
    input_data = request.json.get('input_data')  # 获取用户提交的数据
    inputs = {session.get_inputs()[0].name: input_data}  # 准备输入数据
    output = session.run(None, inputs)[0]  # 执行推理
    return jsonify({'prediction': output.tolist()})  # 返回预测结果


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)