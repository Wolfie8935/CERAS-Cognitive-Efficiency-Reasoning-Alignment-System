try:
    from langchain_openai import ChatOpenAI
    print("SUCCESS: langchain_openai imported")
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
        print("SUCCESS: langchain_community.chat_models imported")
    except ImportError as e:
        print(f"FAILURE: {e}")
except Exception as e:
    print(f"ERROR: {e}")
