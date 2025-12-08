import affinetes as af_env
import asyncio

async def main():
    # Connect to user-deployed environment service
    env = af_env.load_env(
        mode="url",
        base_url="http://localhost:8000"
    )
    
    # Execute methods
    result = await env.evaluate(
        model="demonwizard0/affine-new-3",
        base_url="http://host.docker.internal:30000/v1",
        task_type='sat',
        api_key='random',
        task_id=10
    )
    
    # print(f"Score: {result['score']}")
    print(result)
    
    # Cleanup
    await env.cleanup()

asyncio.run(main())