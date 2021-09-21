import boto3
import json
# from flask import current_app


def set_policy(bucket):
    # Create a bucket policy
    bucket_name = bucket
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AddPerm",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": f"arn:aws:s3:::{bucket_name}/*"
        }]
    }

    # Convert the policy from JSON dict to string
    bucket_policy = json.dumps(bucket_policy)

    # Set the new policy
    s3 = boto3.client("s3")
    s3.put_bucket_policy(Bucket=bucket_name, Policy=bucket_policy)


def S3_upload(BUCKET, current_user, file, key):
    # set_policy(BUCKET)
    s3 = boto3.resource('s3')
    # key = current_user + "-video/" + key + ".mp4"
    try:
        s3.Bucket(BUCKET).put_object(Key=key, Body=file)
        print(f"successfully uploaded {key}!")
    except Exception as e:
        print(e)
    # s3.put_object(Bucket=BUCKET, Key=key, Body=file)


def get_video_url(BUCKET, KEY):
    s3_client = boto3.client("s3")
    response = s3_client.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": KEY})
    return response


def download_file(file_name, current_user, bucket):
    s3 = boto3.resource("s3")
    key = current_user + "-video"
    output = f"downloads/{file_name}"
    s3.meta.client.download_file(bucket, key, output)

    return output


def list_files(bucket, prefix):
    s3 = boto3.client("s3")
    conts = []
    response = s3.list_objects(Bucket=bucket, Prefix=prefix)
    for content in response.get('Contents', []):
        conts.append(content.get('Key'))
        print(content.get('Key'))

    return conts
