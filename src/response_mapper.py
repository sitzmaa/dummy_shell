def map_to_response(terminal_output, model, tokenizer):
    inputs = tokenizer.encode(terminal_output, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
