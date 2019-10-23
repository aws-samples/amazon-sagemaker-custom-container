import os
import boto3
import json

# grab environment variables
SAGEMAKER_ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']
BUCKET_NAME = os.environ['BUCKET_NAME']
runtime= boto3.client('runtime.sagemaker')
s3 = boto3.resource('s3')

def lambda_handler(event, context):
    OBJECT_KEY = event['paper image'] 
    file_name = '/tmp/test_image.jpg'
    s3.Bucket(BUCKET_NAME).download_file(OBJECT_KEY, file_name)

    payload = ''

    with open(file_name, 'rb') as f:
        payload = f.read()
        payload = bytearray(payload)

    response = runtime.invoke_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME,
                                       ContentType='application/x-image',
                                       Body=payload)

    result = json.loads(response['Body'].read().decode())
    print(result)
    pred = result['predictions']['class']

    return pred 