using System;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

/// <summary>
/// Small, dependency-free folder picker for the Unity Editor and Windows builds.
/// </summary>
public static class NativeFolderDialog
{
    public static string Open(string title)
    {
#if UNITY_EDITOR
        return UnityEditor.EditorUtility.OpenFolderPanel(title, string.Empty, string.Empty);
#elif UNITY_STANDALONE_WIN
        return OpenWindowsFolderDialog(title);
#else
        Debug.LogError("Folder selection is currently supported in the Unity Editor and Windows builds.");
        return string.Empty;
#endif
    }

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
    private const uint ReturnOnlyFileSystemDirectories = 0x0001;
    private const uint NewDialogStyle = 0x0040;

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct BrowseInfo
    {
        public IntPtr owner;
        public IntPtr root;
        public IntPtr displayName;
        [MarshalAs(UnmanagedType.LPWStr)] public string title;
        public uint flags;
        public IntPtr callback;
        public IntPtr parameter;
        public int image;
    }

    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    private static extern IntPtr SHBrowseForFolderW(ref BrowseInfo browseInfo);

    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SHGetPathFromIDListW(IntPtr itemIdList, StringBuilder path);

    private static string OpenWindowsFolderDialog(string title)
    {
        IntPtr displayName = Marshal.AllocHGlobal(2048);
        try
        {
            var browseInfo = new BrowseInfo
            {
                displayName = displayName,
                title = title,
                flags = ReturnOnlyFileSystemDirectories | NewDialogStyle
            };

            IntPtr itemIdList = SHBrowseForFolderW(ref browseInfo);
            if (itemIdList == IntPtr.Zero)
            {
                return string.Empty;
            }

            try
            {
                var path = new StringBuilder(1024);
                return SHGetPathFromIDListW(itemIdList, path) ? path.ToString() : string.Empty;
            }
            finally
            {
                Marshal.FreeCoTaskMem(itemIdList);
            }
        }
        finally
        {
            Marshal.FreeHGlobal(displayName);
        }
    }
#endif
}
