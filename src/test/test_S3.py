import pytest
import boto3
from ..application import awsS3 as aws

from moto import mock_s3

BUCKET = "Test_Bucket"


@pytest.fixture()
def moto_boto():
    with mock_s3():
        res = boto3.resource("s3")
        res.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}, )
        yield


def test_creation(moto_boto):
    client = boto3.client("s3")
    assert client.list_objects(Bucket=BUCKET)["Name"] == "Test_Bucket"


def test_upload(moto_boto):
    client = boto3.client("s3")
    with open("static/img/front.png", "rb") as file:
        aws.S3_upload(BUCKET, "test_user", file, 'img.mp4')
    assert client.list_objects(Bucket=BUCKET)["Contents"][0]["Key"] == "img.mp4"


def test_listcontents(moto_boto):
    assert aws.list_files(BUCKET, "test_user") == []
