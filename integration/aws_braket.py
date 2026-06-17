
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AWSBraketIntegration:
    
    def __init__(self, s3_bucket: str, s3_prefix: str):
        try:
            from braket.aws import AwsDevice
            self.AwsDevice = AwsDevice
            self.s3_bucket = s3_bucket
            self.s3_prefix = s3_prefix
        except ImportError:
            raise ImportError("Amazon Braket SDK not installed")
    
    def list_available_devices(self) -> List[Dict[str, Any]]:
        try:
            devices = self.AwsDevice.get_devices(
                provider="Amazon",
                status="AVAILABLE"
            )
            
            result = []
            for device in devices:
                result.append({
                    'device_arn': device.arn,
                    'name': device.name,
                    'provider': device.provider_name,
                    'status': device.status,
                    'device_type': device.device_type
                })
            
            return result
        except Exception as e:
            logger.error(f"Failed to list devices: {e}")
            return []
    
    def submit_to_device(self, circuit: Any, device_arn: str, 
                        shots: int = 100) -> str:
        try:
            device = self.AwsDevice(device_arn)
            
            s3_destination_folder = (self.s3_bucket, self.s3_prefix)
            
            task = device.run(
                circuit,
                s3_destination_folder=s3_destination_folder,
                shots=shots
            )
            
            return task.id
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            return None
    
    def get_task_result(self, task_arn: str) -> Dict[str, Any]:
        try:
            from braket.aws import AwsDevice
            
            task = AwsDevice.retrieve_job(task_arn)
            
            if task.state != 'COMPLETED':
                return {'state': task.state, 'ready': False}
            
            result = task.result()
            
            return {
                'task_arn': task_arn,
                'result': result,
                'state': task.state,
                'ready': True
            }
        except Exception as e:
            logger.error(f"Failed to retrieve task: {e}")
            return {'error': str(e)}
