import asyncio

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.generic import Generic as GenericComponent
from viam.services.generic import Generic as GenericService
from real_connect import real_connect, grip_name


async def hold_cup(gripper):
    # await gripper.open()
    await gripper.do_command({'set_gripper_torque': 50})


async def test_cup(gripper):
    await gripper.grab()
    result = await gripper.do_command({'get_gripper_torque': True})
    print(result)


async def main():
    async with await real_connect() as machine:
        print('Resources:')
        print(machine.resource_names)
        
        
        gripper = Gripper.from_robot(machine, grip_name)
        hold = await gripper.is_holding_something()
        print('Gripper holding something: ', hold.is_holding_something)

        # await gripper.grab()
        # await gripper.open()
        # await gripper.close()

        # await hold_cup(gripper)
        await test_cup(gripper)

        hold = await gripper.is_holding_something()
        print('Gripper holding something: ', hold.is_holding_something)



if __name__ == '__main__':
    asyncio.run(main())
