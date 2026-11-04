"""Observe actual archive requests without changing the archive protocol."""

class MeasuredStream:
    def __init__(self,stream):
        self.stream=stream
        self.measurement={'archive_chunk_requests':0,'archive_raw_payload_bytes_downloaded':0,
            'archive_ACK_requests':0,'archive_ACK_original_bytes':0}
    def request(self,operation,*args,**kwargs):
        result=self.stream.request(operation,*args,**kwargs)
        if operation=='archive_chunk':
            self.measurement['archive_chunk_requests']+=1
            self.measurement['archive_raw_payload_bytes_downloaded']+=result['raw_bytes']
        if operation=='archive_ack':
            self.measurement['archive_ACK_requests']+=1
            self.measurement['archive_ACK_original_bytes']+=args[0]['ack']['bytes']
        return result
    def close(self):
        return self.stream.close()
