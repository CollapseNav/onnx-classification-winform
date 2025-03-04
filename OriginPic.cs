namespace onnx_classification_winform;

public partial class OriginPic : Form
{
    private PictureBox Picture = new PictureBox();

    public OriginPic()
    {
        InitializeComponent();
        Controls.Add(Picture);
    }

    public void LoadImage(Bitmap bitmap)
    {
        Size = new Size(bitmap.Width, 39 + bitmap.Height);
        Picture.Size = new Size(bitmap.Width, bitmap.Height);
        Picture.Image = bitmap;
    }
}
