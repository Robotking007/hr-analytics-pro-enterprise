"""
TensorFlow Fix for Windows DLL Issues
"""
import subprocess
import sys
import os

def fix_tensorflow_windows():
    """Fix TensorFlow DLL issues on Windows"""
    print("🔧 Fixing TensorFlow DLL issues on Windows...")
    
    commands = [
        # Uninstall problematic TensorFlow versions
        [sys.executable, "-m", "pip", "uninstall", "tensorflow", "tensorflow-intel", "-y"],
        
        # Install Visual C++ Redistributable compatible version
        [sys.executable, "-m", "pip", "install", "tensorflow-cpu==2.10.0"],
        
        # Install compatible numpy
        [sys.executable, "-m", "pip", "install", "numpy==1.21.6"],
        
        # Install Microsoft Visual C++ Redistributable if needed
        # User will need to do this manually from: https://aka.ms/vs/17/release/vc_redist.x64.exe
    ]
    
    for i, cmd in enumerate(commands):
        try:
            print(f"Step {i+1}: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Step {i+1} completed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Step {i+1} failed: {e}")
            if i < 2:  # Continue for uninstall steps
                continue
            else:
                return False
    
    print("\n📋 Manual Step Required:")
    print("Download and install Microsoft Visual C++ Redistributable:")
    print("https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("Then restart your computer.")
    
    return True

def test_tensorflow():
    """Test if TensorFlow works"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} loaded successfully!")
        
        # Test basic operation
        test_tensor = tf.constant([1, 2, 3, 4])
        print(f"✅ TensorFlow operations working: {test_tensor}")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TensorFlow Windows Fix")
    print("=" * 40)
    
    if fix_tensorflow_windows():
        print("\n🧪 Testing TensorFlow...")
        test_tensorflow()
    else:
        print("❌ Fix failed")
