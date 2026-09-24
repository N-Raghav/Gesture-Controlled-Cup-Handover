from viam.robot.client import RobotClient
import os
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv('PROD_API_KEY')
API_KEY_ID = os.getenv('PROD_API_KEY_ID')


async def real_connect():
    opts = RobotClient.Options.with_api_key(
        api_key=API_KEY,    #type: ignore
        api_key_id=API_KEY_ID   #type: ignore
    )
    
    return await RobotClient.at_address('armfarm8-main.310sld03v2.viam.cloud', opts)


grip_name = 'gripper'
