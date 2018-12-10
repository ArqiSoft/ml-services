import os

from mass_transit.MTMessageProcessor import PureConsumer
from mass_transit.mass_transit_constants import BLOB_LOADED_TEST

TEMP_FOLDER = os.environ['OSDR_TEMP_FILES_FOLDER']


def blob_loaded(body):
    blob_id = body['BlobInfo']['id']
    bucket = body['BlobInfo']['bucket']
    parent_id = body['BlobInfo']['metadata']['parentId']
    file_name = '{}/{}'.format(TEMP_FOLDER, parent_id)
    data = str({'blob_id': blob_id, 'parent_id': parent_id})
    file_to_write = open(file_name, 'a')
    file_to_write.write('{}\n'.format(data))
    file_to_write.close()

    return None


if __name__ == '__main__':
    BLOB_LOADED_TEST['event_callback'] = blob_loaded
    blob_loaded_test_consumer = PureConsumer(
        BLOB_LOADED_TEST, infinite_consuming=True)
    blob_loaded_test_consumer.start_consuming()
