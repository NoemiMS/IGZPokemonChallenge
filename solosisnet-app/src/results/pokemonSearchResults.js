import React from 'react';
import PokemonResultRow from './pokemonResultRow';
import './EmojiResults.css';

class EmojiResults extends React.Component {
  render() {
    return (
      <div className="component-emoji-results">
        {
          this.props.emojiData.map((emojiData) => {
            return (
              <PokemonResultRow
                title={emojiData.title}
              />
            );
          })
        }
      </div>
    );
  }
}
EmojiResults.propTypes = {
  emojiData: React.PropTypes.array,
};
export default EmojiResults;
