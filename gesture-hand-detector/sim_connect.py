from viam.robot.client import RobotClient
from dotenv import load_dotenv
import os


load_dotenv()
API_KEY = os.getenv('MY_API_KEY')
API_KEY_ID = os.getenv('MY_API_KEY_ID')


async def sim_connect():
    opts = RobotClient.Options.with_api_key(        
        api_key=API_KEY,    #type: ignore
        api_key_id=API_KEY_ID   #type: ignore
    )
    
    return await RobotClient.at_address('viam-test-main.xlzng6bp04.viam.cloud', opts)


grip_name = 'gripper-1'