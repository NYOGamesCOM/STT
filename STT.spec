# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['stt.py'],
    pathex=[],
    binaries=[],
    datas=[('config.json', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='STT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',
        'python*.dll',
        '_multiarray_umath.*',
        '_multiarray_tests.*',
        'libopenblas*.dll',
        'libiomp*.dll',
        'mkl_*.dll',
        'ctranslate2.dll',
        'onnxruntime*.dll',
        'torch*.dll',
        'cudnn*.dll',
        'cublas*.dll',
    ],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
