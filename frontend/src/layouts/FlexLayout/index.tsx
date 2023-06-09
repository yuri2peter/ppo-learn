import React, { useMemo } from 'react';
import { use100vh } from 'react-div-100vh';
import { useWatchSize } from './sections/useWatchSize';
import { useResizeDetector } from 'react-resize-detector';
import { ContextProvider } from './sections/context';
import ParentBox from './sections/ParentBox';
import FullScreenContent from './sections/FullScreenContent';
import WindowContent from './sections/WindowContent';

/**
 * 万能弹性布局器
 */
const FlexLayout: React.FC<{}> = () => {
  const height100vh = use100vh();
  const { width, height, ref } = useResizeDetector();
  const content = useWatchSize({
    width: width || 0,
    height: height || 0,
  });
  const { mode } = content;
  const contentEl = useMemo(() => {
    let el = null;
    if (['MOBILE', 'DESKTOP'].includes(mode)) {
      el = <FullScreenContent />;
    }
    if (['WINDOW'].includes(mode)) {
      el = <WindowContent />;
    }
    return el;
  }, [mode]);
  return (
    <ContextProvider value={content}>
      <ParentBox sizerRef={ref} height={height100vh || 0}>
        {contentEl}
      </ParentBox>
    </ContextProvider>
  );
};

export default FlexLayout;
