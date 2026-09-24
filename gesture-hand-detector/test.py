import asyncio

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.generic import Generic as GenericComponent
from viam.services.generic import Generic as GenericService
from sim_connect import sim_connect, grip_name


async def main():
    async with await sim_connect() as machine:
        print('Resources:')
        print(machine.resource_names)
        
        # arm-1
        arm_1 = Arm.from_robot(machine, "arm-1")
        arm_1_return_value = await arm_1.get_end_position()
        print(f"arm-1 get_end_position return value: {arm_1_return_value}")

        # gripper-1
        gripper_1 = Gripper.from_robot(machine, "gripper-1")
        gripper_1_return_value = await gripper_1.is_moving()
        print(f"gripper-1 is_moving return value: {gripper_1_return_value}")

        # Note that the following block is commented out because it may actuate
        # or because its argument semantics are unknown. Use with caution.
        # floor
        # floor = GenericComponent.from_robot(machine, "floor")
        # floor_return_value = await floor.do_command({})
        # print(f"floor do_command return value: {floor_return_value}")

if __name__ == '__main__':
    asyncio.run(main())
