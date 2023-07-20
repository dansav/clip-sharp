using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ClipSharp.Example.Wpf
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly MainViewModel _vm;

        public MainWindow()
        {
            InitializeComponent();
            AllowDrop = true;
            DataContext = _vm = new MainViewModel();
            Loaded += async (sender, args) =>
            {
                try
                {
                    await Task.Run(() =>
                    {
                        _vm.Initialize();
                    });
                }
                catch (Exception e)
                {
                    MessageBox.Show(e.ToString());
                    Application.Current.Shutdown();
                    return;
                }

                IsEnabled = true;
            };
            IsEnabled = false;
        }

        protected override void OnDrop(DragEventArgs e)
        {
            base.OnDragEnter(e);

            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);
                _vm.HandleFileDrop(files);

            }
        }
    }
}
